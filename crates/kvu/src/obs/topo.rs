//! Topologie SU(2) (WP19, tranche 3) : charge améliorée (trèfles m×m), cooling corrigé, flot multi-temps.
//! Propriétaire : WP19. Définitions gelées : `GATES.md` v4.3 (+ avenant v4.3.1).
//!
//! Les types de sortie sont figés : le format JSON d'une ligne de `topo.jsonl` en dépend.

use super::ref_cpu::{
    en_champ, en_quaternions, pas_rk3, plaquette_site, produit_chemin, Reseau, INV_2PI2, PLANS, UN,
};
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{
    k_final, k_reduce, philox4x32_10, qadd, qdag, qmul, qnorm, qscale, u01, Geo, Q,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub mod gpu;

/// Coefficients de la charge améliorée pour les trèfles m×m, m = 1, 2, 3 (v4.3 : Vandermonde en m²).
pub const COEFS_AMELIORE: [f64; 3] = [3.0 / 2.0, -3.0 / 5.0, 1.0 / 10.0];

/// Temps de flot t/a² mesurés (INFO sauf t/a² = 3 : TOPO-Q).
pub const TEMPS_A2: [f64; 4] = [1.0, 2.0, 3.0, 5.0];
/// Multiples de t₀ mesurés (INFO).
pub const MULT_T0: [f64; 4] = [1.0, 2.0, 3.0, 5.0];
/// Indice de t/a² = 3 dans `TEMPS_A2`.
pub const IDX_T3: usize = 2;

/// Paramètres d'une mesure topologique complète d'une configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParamsTopo {
    /// Pas du flot RK3 (v4.3 : 0,01).
    pub dt: f64,
    /// Nombre de balayages de cooling (v4.3 : 20).
    pub balayages: usize,
    /// Cible de t²E_plaq pour t₀ (0,3).
    pub cible_t2e: f64,
    /// Nombre maximal de pas de flot si t₀ n'est pas trouvé (5000 ⇒ t = 50).
    pub pas_max: usize,
    /// Flot complet (t/a² = 1, 2, 3, 5 et k·t₀) ou partiel (jusqu'à t/a² = 3 : 3 valeurs, t₀ non mesuré) : avenant v4.3.1 §B.
    pub flot_complet: bool,
    /// Mesure SANS flot (cooling seul, sur la copie brute) : `flot_a2` vide, `t0` et `flot_t0` absents.
    /// Avenant v5.WP19 : TOPO-LIT-V5 ne lit que le cooling. Absent d'un JSON ancien : `false`.
    #[serde(default)]
    pub sans_flot: bool,
}

impl Default for ParamsTopo {
    fn default() -> Self {
        ParamsTopo {
            dt: 0.01,
            balayages: 20,
            cible_t2e: 0.3,
            pas_max: 5000,
            flot_complet: true,
            sans_flot: false,
        }
    }
}

/// Série le long du cooling : indice 0 = avant tout balayage, indice k = après k balayages (longueur balayages + 1).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SerieCooling {
    /// Charge améliorée Q (non arrondie).
    pub q_imp: Vec<f64>,
    /// Charge trèfle non améliorée (m = 1).
    pub q_tre: Vec<f64>,
    /// Action de Wilson S = Σ_{x, μ<ν} (1 − ½ Re Tr P_μν(x)) = 6V − Σ_x plaquette_site(x).
    pub action: Vec<f64>,
}

/// Mesures au long du flot aux quatre temps t/a² = 1, 2, 3, 5 (`TEMPS_A2`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FlotA2 {
    /// E_plaq = 24 (1 − ⟨P⟩) à chaque temps.
    pub e_plaq: Vec<f64>,
    pub q_tre: Vec<f64>,
    pub q_imp: Vec<f64>,
}

/// Mesures aux temps k·t₀ (k = 1, 2, 3, 5), au premier pas de grille ≥ k·t₀ : n_k = ⌈k·t₀/dt − 1e-9⌉.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FlotT0 {
    /// Temps de flot effectif n_k·dt.
    pub t_eff: Vec<f64>,
    pub q_tre: Vec<f64>,
    pub q_imp: Vec<f64>,
}

/// Une ligne de `topo.jsonl` : toutes les mesures topologiques d'une configuration.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RecordTopo {
    /// Nom de l'ensemble (`q`, `lit`, `lit_mut`, `obs4`…), fourni par `--ens`.
    pub ens: String,
    /// Numéro de chaîne (1…), fourni par `--chaine`.
    pub chaine: usize,
    /// Numéro de configuration dans la chaîne (1…), lu dans `cfg_NNNNNN.npy`.
    pub cfg: usize,
    /// Backend de génération (`hip`, `cuda`, `cpu`), fourni par `--backend-gen`.
    pub backend_gen: String,
    /// Backend de la mesure (`hip`, `cuda`, `cpu`).
    pub backend_mesure: String,
    /// Chemin du fichier de configuration lu.
    pub fichier: String,
    /// Côté du réseau.
    pub l: usize,
    /// ⟨½ Re Tr U_p⟩ de la configuration brute (= 1 − E_plaq(0)/24).
    pub plaq0: f64,
    /// t₀ de la configuration (croisement de t²E_plaq = cible, interpolation linéaire de `flot::echelles`), None si non trouvé.
    pub t0: Option<f64>,
    pub flot_a2: FlotA2,
    /// None si t₀ non trouvé.
    pub flot_t0: Option<FlotT0>,
    pub cooling: SerieCooling,
}

/// Mesure complète d'une configuration sur le CPU (référence) : flot RK3 (copie), cooling (copie), charges.
/// Définie par `GATES.md` v4.3 et l'avenant v4.3.1 ; écrite par le lot CPU.
pub fn mesurer_cpu(champ: &ChampHote, p: &ParamsTopo) -> Resultat<RecordTopo> {
    let geo = Reseau::nouveau(champ)?.geo;
    if !p.dt.is_finite()
        || p.dt <= 0.0
        || !p.cible_t2e.is_finite()
        || p.cible_t2e <= 0.0
        || p.pas_max < 500
    {
        return Err(Erreur::Config(
            "topologie : dt et cible finis > 0, pas_max ≥ 500 requis".into(),
        ));
    }
    let pas_a2 = TEMPS_A2.map(|t| (t / p.dt).round());
    if pas_a2.iter().any(|&n| n < 1.0 || n > p.pas_max as f64) {
        return Err(Erreur::Config(
            "topologie : les temps TEMPS_A2 doivent être accessibles avant pas_max".into(),
        ));
    }
    let pas_a2 = pas_a2.map(|n| n as usize);
    // Flot partiel : seulement t/a² = 1, 2, 3 (pas 100, 200, 300) ; t₀ non mesuré (avenant v4.3.1 §B).
    let pas_mesures: Vec<usize> = if p.flot_complet {
        pas_a2.to_vec()
    } else {
        pas_a2[..=IDX_T3].to_vec()
    };
    let mut record = RecordTopo {
        backend_mesure: "cpu".into(),
        l: champ.l,
        plaq0: charges(champ)?.plaq,
        ..RecordTopo::default()
    };
    if !p.sans_flot {
        let mut u = en_quaternions(champ);
        let mut precedent = (0.0, 0.0);
        let mut pas_t0 = None;
        let mut arret = if p.flot_complet { p.pas_max } else { pas_a2[IDX_T3] };
        let mut n = 0;
        loop {
            n += 1;
            pas_rk3(&mut u, &geo, p.dt);
            let courant = en_champ(&u, champ.l);
            let res = Reseau::nouveau(&courant)?;
            let e_plaq = 24.0 * (1.0 - somme_plaquettes(&res) / (6.0 * geo.v as f64));
            let t = n as f64 * p.dt;
            let t2e = t * t * e_plaq;
            if p.flot_complet && record.t0.is_none() && precedent.1 < p.cible_t2e && t2e >= p.cible_t2e {
                // Même interpolation et même garde sur le dénominateur que flot::echelles.
                let denom = t2e - precedent.1;
                let t0 = if denom.abs() < 1e-300 {
                    precedent.0
                } else {
                    precedent.0 + (p.cible_t2e - precedent.1) / denom * (t - precedent.0)
                };
                let grille = MULT_T0.map(|k| (k * t0 / p.dt - 1e-9).ceil().max(n as f64));
                if grille
                    .iter()
                    .any(|x| !x.is_finite() || *x >= usize::MAX as f64)
                {
                    return Err(Erreur::Config(
                        "topologie : temps k·t₀ hors de la grille représentable".into(),
                    ));
                }
                let grille = grille.map(|x| x as usize);
                arret = 500.max(pas_a2[3]).max(grille[3]);
                pas_t0 = Some(grille);
                record.t0 = Some(t0);
                record.flot_t0 = Some(FlotT0::default());
            }
            let mesure_t0 = pas_t0.is_some_and(|grille| grille.contains(&n));
            if pas_mesures.contains(&n) || mesure_t0 {
                let c = charges(&courant)?;
                for &pas in &pas_mesures {
                    if pas == n {
                        record.flot_a2.e_plaq.push(e_plaq);
                        record.flot_a2.q_tre.push(c.q_tre);
                        record.flot_a2.q_imp.push(c.q_imp);
                    }
                }
                if let (Some(grille), Some(flot)) = (pas_t0, record.flot_t0.as_mut()) {
                    for pas in grille {
                        if pas == n {
                            flot.t_eff.push(t);
                            flot.q_tre.push(c.q_tre);
                            flot.q_imp.push(c.q_imp);
                        }
                    }
                }
            }
            precedent = (t, t2e);
            if n >= arret {
                break;
            }
        }
    }
    record.cooling = cooling(champ, p.balayages, true)?.0;
    Ok(record)
}

/// Les quatre orientations positives du trèfle, comme dans la référence CPU.
fn feuilles(mu: usize, nu: usize) -> [[(usize, i32); 4]; 4] {
    [
        [(mu, 1), (nu, 1), (mu, -1), (nu, -1)],
        [(nu, 1), (mu, -1), (nu, -1), (mu, 1)],
        [(mu, -1), (nu, -1), (mu, 1), (nu, 1)],
        [(nu, -1), (mu, 1), (nu, 1), (mu, -1)],
    ]
}

/// Trèfle de quatre boucles m×m, chaque côté répétant m fois le même pas.
pub fn trefle_mxm(res: &Reseau, s: usize, m: usize) -> [[f64; 3]; 6] {
    assert!(m > 0, "trèfle : m doit être positif");
    let mut g = [[0.0; 3]; 6];
    for (p, &(mu, nu)) in PLANS.iter().enumerate() {
        let mut somme = [0.0; 4];
        for feuille in feuilles(mu, nu) {
            let chemin: Vec<_> = feuille
                .into_iter()
                .flat_map(|pas| std::iter::repeat_n(pas, m))
                .collect();
            somme = qadd(somme, produit_chemin(res, s, &chemin));
        }
        g[p] = [0.25 * somme[1], 0.25 * somme[2], 0.25 * somme[3]];
    }
    g
}

fn combiner_champs(g1: [[f64; 3]; 6], g2: [[f64; 3]; 6], g3: [[f64; 3]; 6]) -> [[f64; 3]; 6] {
    let mut g = [[0.0; 3]; 6];
    for (p, plan) in g.iter_mut().enumerate() {
        for (k, x) in plan.iter_mut().enumerate() {
            *x = COEFS_AMELIORE[0] * g1[p][k] / 1.0
                + COEFS_AMELIORE[1] * g2[p][k] / 4.0
                + COEFS_AMELIORE[2] * g3[p][k] / 9.0;
        }
    }
    g
}

/// Champ amélioré, avec l'ordre flottant c₁g₁/1 + c₂g₂/4 + c₃g₃/9 gelé.
pub fn champ_ameliore(res: &Reseau, s: usize) -> [[f64; 3]; 6] {
    combiner_champs(
        trefle_mxm(res, s, 1),
        trefle_mxm(res, s, 2),
        trefle_mxm(res, s, 3),
    )
}

/// Densité topologique ; même accumulation (+, −, +) que densite_trefle_site.
pub fn densite_q(g: &[[f64; 3]; 6]) -> f64 {
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mut q = 0.0;
    q += dot(g[0], g[5]);
    q += -dot(g[1], g[4]);
    q += dot(g[2], g[3]);
    INV_2PI2 * q
}

/// Plaquette moyenne et charges totales, non arrondies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Charges {
    pub plaq: f64,
    pub q_tre: f64,
    pub q_imp: f64,
}

/// Les trois observables en un passage parallèle, puis réductions à arbre fixe.
pub fn charges(champ: &ChampHote) -> Resultat<Charges> {
    let res = Reseau::nouveau(champ)?;
    let par_site: Vec<[f64; 3]> = (0..res.geo.v)
        .into_par_iter()
        .map(|s| {
            let g1 = trefle_mxm(&res, s, 1);
            let g = combiner_champs(g1, trefle_mxm(&res, s, 2), trefle_mxm(&res, s, 3));
            [plaquette_site(&res, s), densite_q(&g1), densite_q(&g)]
        })
        .collect();
    let somme = |k| {
        let valeurs: Vec<f64> = par_site.iter().map(|x| x[k]).collect();
        k_final(&k_reduce(&valeurs), 1.0)
    };
    Ok(Charges {
        plaq: somme(0) / (6.0 * res.geo.v as f64),
        q_tre: somme(1),
        q_imp: somme(2),
    })
}

/// Charge améliorée totale.
pub fn charge_amelioree(champ: &ChampHote) -> Resultat<f64> {
    Ok(charges(champ)?.q_imp)
}

/// Charge trèfle totale, identique au bit à la référence CPU.
pub fn charge_trefle(champ: &ChampHote) -> Resultat<f64> {
    Ok(charges(champ)?.q_tre)
}

/// Mutant TOPO-Q-MUT : ajouter 1/(2V) à chaque densité décale la charge de ½.
/// L'ajout est fait sur la charge totale afin d'éviter les arrondis de la réduction.
pub fn decaler_charge(q: f64) -> f64 {
    q + 0.5
}

fn voisin(geo: &Geo, s: usize, d: usize, sens: i32) -> usize {
    let xd = geo.coords(s)[d];
    if sens > 0 {
        geo.nb_up(s, d, xd)
    } else {
        geo.nb_dn(s, d, xd)
    }
}

/// W, somme des six chemins ouverts de x+μ vers x, indépendante du staple historique.
pub fn w_agrafes(u: &[Q], geo: &Geo, mu: usize, s: usize) -> Q {
    let lien = |d, x| u[d * geo.v + x];
    let smu = voisin(geo, s, mu, 1);
    let mut w = [0.0; 4];
    for nu in 0..4 {
        if nu == mu {
            continue;
        }
        let snu = voisin(geo, s, nu, 1);
        let sdn = voisin(geo, s, nu, -1);
        let smudn = voisin(geo, smu, nu, -1);
        let haut = qmul(qmul(lien(nu, smu), qdag(lien(mu, snu))), qdag(lien(nu, s)));
        let bas = qmul(
            qmul(qdag(lien(nu, smudn)), qdag(lien(mu, sdn))),
            lien(nu, sdn),
        );
        w = qadd(w, qadd(haut, bas));
    }
    w
}

/// Maximisation de Re Tr(U W) ; W nul laisse libre U, choisi identité.
/// `dagger = false` reproduit le mutant sans conjugaison du 08/10.
pub fn maj_cooling(w: Q, dagger: bool) -> Q {
    let norme = qnorm(w);
    if norme == 0.0 {
        UN
    } else {
        qscale(if dagger { qdag(w) } else { w }, 1.0 / norme)
    }
}

/// Un balayage dans l'ordre (μ, parité), avec lecture puis écriture de chaque classe.
pub fn balayage_cooling(u: &mut [Q], geo: &Geo, dagger: bool) {
    assert!(geo.l.is_multiple_of(2), "cooling : L pair requis");
    for mu in 0..4 {
        for parite in 0..2 {
            let nouveaux: Vec<(usize, Q)> = (0..geo.v / 2)
                .into_par_iter()
                .map(|i| {
                    let s = geo.site_of(i, parite).0;
                    (s, maj_cooling(w_agrafes(u, geo, mu, s), dagger))
                })
                .collect();
            for (s, q) in nouveaux {
                u[mu * geo.v + s] = q;
            }
        }
    }
}

fn somme_plaquettes(res: &Reseau) -> f64 {
    let par_site: Vec<f64> = (0..res.geo.v)
        .into_par_iter()
        .map(|s| plaquette_site(res, s))
        .collect();
    k_final(&k_reduce(&par_site), 1.0)
}

/// Action S = 6V − Σ_x plaquette_site, avec réduction à arbre fixe.
pub fn action_wilson(u: &[Q], geo: &Geo) -> f64 {
    let champ = en_champ(u, geo.l);
    let res = Reseau::nouveau(&champ).expect("action de Wilson : champ incohérent");
    6.0 * geo.v as f64 - somme_plaquettes(&res)
}

/// Cooling d'une copie du champ brut ; indice 0 avant tout balayage.
pub fn cooling(
    champ: &ChampHote,
    balayages: usize,
    dagger: bool,
) -> Resultat<(SerieCooling, ChampHote)> {
    let geo = Reseau::nouveau(champ)?.geo;
    if !geo.l.is_multiple_of(2) {
        return Err(Erreur::Config(
            "cooling : L pair requis pour le damier".into(),
        ));
    }
    let mut u = en_quaternions(champ);
    let mut courant = champ.clone();
    let mut serie = SerieCooling::default();
    for k in 0..=balayages {
        if k > 0 {
            balayage_cooling(&mut u, &geo, dagger);
            courant = en_champ(&u, geo.l);
        }
        let c = charges(&courant)?;
        serie.q_imp.push(c.q_imp);
        serie.q_tre.push(c.q_tre);
        serie.action.push(action_wilson(&u, &geo));
    }
    Ok((serie, courant))
}

/// Générateur instrumental : clé (bas, haut) de la graine, compteur initial 0.
/// Chaque uniforme consomme un bloc Philox4x32-10 de compteur [bas, haut, 0, 0],
/// incrémente le compteur u64 (modulo 2⁶⁴) et utilise u01(bloc[0], bloc[1]).
/// Chaque normale consomme deux uniformes, par Box–Muller (cosinus, sans cache).
/// Chaque Haar consomme quatre normales puis normalise leur vecteur euclidien.
#[derive(Clone, Debug)]
pub struct AleaPhilox {
    cle: [u32; 2],
    compteur: u64,
}

impl AleaPhilox {
    pub fn nouveau(graine: u64) -> Self {
        Self {
            cle: [graine as u32, (graine >> 32) as u32],
            compteur: 0,
        }
    }

    /// Uniforme strictement dans ]0,1[, sur 52 bits.
    pub fn uniforme(&mut self) -> f64 {
        let bloc = philox4x32_10(
            [self.compteur as u32, (self.compteur >> 32) as u32, 0, 0],
            self.cle[0],
            self.cle[1],
        );
        self.compteur = self.compteur.wrapping_add(1);
        u01(bloc[0], bloc[1])
    }

    pub fn normale(&mut self) -> f64 {
        (-2.0 * self.uniforme().ln()).sqrt() * (std::f64::consts::TAU * self.uniforme()).cos()
    }

    pub fn haar_su2(&mut self) -> Q {
        let q = [
            self.normale(),
            self.normale(),
            self.normale(),
            self.normale(),
        ];
        qscale(q, 1.0 / qnorm(q))
    }
}

/// Flux abélien n₀₁ = n₂₃ = 1, plongé selon σ₃, avec les liens de raccord périodiques.
pub fn champ_abelien(l: usize) -> ChampHote {
    let mut champ = ChampHote::froid(l).expect("champ abélien : L pair ≥ 2 requis");
    let geo = Geo::new(l);
    let angle = |theta: f64| [theta.cos(), 0.0, 0.0, theta.sin()];
    for s in 0..geo.v {
        let x = geo.coords(s);
        for (mu, nu) in [(0, 1), (2, 3)] {
            let q = angle(std::f64::consts::TAU / (l * l) as f64 * x[mu] as f64);
            champ.q[4 * (nu * geo.v + s)..4 * (nu * geo.v + s + 1)].copy_from_slice(&q);
            if x[mu] == l - 1 {
                let q = angle(-std::f64::consts::TAU / l as f64 * x[nu] as f64);
                champ.q[4 * (mu * geo.v + s)..4 * (mu * geo.v + s + 1)].copy_from_slice(&q);
            }
        }
    }
    champ
}

/// Jauge de Haar indépendante à chaque site, tirée dans l'ordre des sites.
pub fn transformer_jauge(champ: &ChampHote, graine: u64) -> ChampHote {
    let geo = Geo::new(champ.l);
    let mut alea = AleaPhilox::nouveau(graine);
    let g: Vec<Q> = (0..geo.v).map(|_| alea.haar_su2()).collect();
    let u = en_quaternions(champ);
    let out: Vec<Q> = (0..4 * geo.v)
        .into_par_iter()
        .map(|li| {
            let (mu, s) = (li / geo.v, li % geo.v);
            qmul(qmul(g[s], u[li]), qdag(g[voisin(&geo, s, mu, 1)]))
        })
        .collect();
    en_champ(&out, champ.l)
}

/// Réflexion x₃ → −x₃ ; le lien selon 3 est reculé et conjugué.
pub fn reflechir_axe3(champ: &ChampHote) -> ChampHote {
    let geo = Geo::new(champ.l);
    let u = en_quaternions(champ);
    let out: Vec<Q> = (0..4 * geo.v)
        .into_par_iter()
        .map(|li| {
            let (mu, s) = (li / geo.v, li % geo.v);
            let x3 = s % geo.l;
            let r = s - x3 + (geo.l - x3) % geo.l;
            if mu == 3 {
                qdag(u[mu * geo.v + voisin(&geo, r, 3, -1)])
            } else {
                u[mu * geo.v + r]
            }
        })
        .collect();
    en_champ(&out, champ.l)
}

/// Matrices instrumentales W = Σ₁⁶ rₖ Vₖ ; tirages dans l'ordre Vₖ puis rₖ.
pub fn cool_id_matrices(graine: u64, n: usize) -> Vec<Q> {
    let mut alea = AleaPhilox::nouveau(graine);
    (0..n)
        .map(|_| {
            let mut w = [0.0; 4];
            for _ in 0..6 {
                let v = alea.haar_su2();
                w = qadd(w, qscale(v, alea.uniforme()));
            }
            w
        })
        .collect()
}

/// Écart à Re Tr(U W) = 2|W|. Le scalaire de U W est U·conj(W), et non U·W.
pub fn cool_id_ecart(w: &[Q], dagger: bool) -> f64 {
    w.iter()
        .map(|&w| {
            let u = maj_cooling(w, dagger);
            let wd = qdag(w);
            let trace = 2.0 * (u[0] * wd[0] + u[1] * wd[1] + u[2] * wd[2] + u[3] * wd[3]);
            (trace - 2.0 * qnorm(w)).abs()
        })
        .fold(0.0, f64::max)
}
