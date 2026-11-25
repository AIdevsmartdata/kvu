//! [EXPÉRIMENTAL] Secteurs de flux (1,1,0) et (1,1,1) de `spectro gpu-torelon`, à impulsion totale nulle.
//! Référence CPU F64 et assemblage hôte pour `k_flux` et `k_flux_111` (`noyaux/su2_spectro.cu`).
//!
//! Secteur à deux enroulements. Pour les directions d1, d2 (côtés égaux L), un pas de superlien ℓ (1 en APE, 2^b en blocage) et un
//! point de base x, avec L_i(x) = ∏_{k<L/ℓ} U_i(x + kℓ ê_i) (ligne NON tracée) :
//!   a+ = ½Tr[L1 L2]   a− = ½Tr[L1 L2†]     (lacet en L, déplacements (L,±L))
//!   b+ = ½Tr ∏_{k<L/ℓ} [U1(x_k) U2(x_k+ℓê1)],  x_{k+1} = x_k + ℓ(ê1+ê2)      (escalier, déplacement (L,+L))
//!   b− = ½Tr ∏_{k<L/ℓ} [U1(x_k) U2(x_k+ℓê1−ℓê2)†], x_{k+1} = x_k + ℓ(ê1−ê2)  (déplacement (L,−L))
//!   T1 = ½Tr L1(x), T2 = ½Tr L2(x)
//! Les sommes S_q(t) = Σ_{x : x_temps = t} (toute la tranche) sont les sorties du noyau. Opérateurs rapportés, par niveau :
//! `[a+, a−, b+, b−, c, T1, T2]` avec c = S_T1 · S_T2 (produit de deux torelons d'axe, somme sur la tranche). La base de
//! corrélation (par niveau) est `[a+ + a−, b+ + b−, c, T1, T2]` (paire sous la réflexion x₂ → −x₂).
//!
//! Secteur (1,1,1). Pour les axes spatiaux a < b < c de côté L, on moyenne les ½Tr des escaliers positifs
//! `(a,b,c)^(L/ℓ)` et `(a,c,b)^(L/ℓ)` puis on somme sur les points de base de la tranche. Chaque chemin parcourt
//! L dans chaque direction et se ferme sur le tore. Les permutations cycliques reviennent à translater le point de base
//! et à permuter cycliquement les facteurs sous la trace. Les deux ordres couvrent donc la symétrie des permutations
//! des axes de la somme de tranche. La projection utilise les enroulements positifs et leurs inverses implicites
//! par la trace SU(2), sans moyenner les huit choix de signe de l'enroulement.

use super::bloc::{champs_bloques, decaler, trace_chemin_l, ParamBloc};
use super::gpu_torelon::ParamTorelon;
use super::{lisser, Mutant};
use crate::commun::{Erreur, Resultat};
use crate::heritage::su2_hb::{qdag, qmul, Q};
use crate::su2::tube::reseau::GeoA;

/// nombre d'opérateurs rapportés par niveau (`ops_flux.bin`)
pub const NOPS_BRUTS: usize = 7;
/// nombre d'opérateurs de la base de corrélation par niveau
pub const NOPS_BASE: usize = 5;
pub const NOMS_BRUTS: [&str; NOPS_BRUTS] = ["a+", "a-", "b+", "b-", "c=T1*T2", "T1", "T2"];
pub const NOMS_BASE: [&str; NOPS_BASE] = ["a++a-", "b++b-", "c=T1*T2", "T1", "T2"];

/// Directions du secteur à deux enroulements (d1 < d2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SecteurFlux {
    pub d: [usize; 2],
}

impl SecteurFlux {
    /// `texte` = triplet (axes spatiaux) ou quadruplet (axes 0..3) avec deux charges unitaires.
    pub fn analyser(texte: &str, cotes: &[usize; 4], temps: usize) -> Resultat<Self> {
        match SecteurCentre::analyser(texte, cotes, temps)? {
            SecteurCentre::Deux(f) => Ok(f),
            SecteurCentre::Trois(_) => Err(Erreur::Config("--secteur-flux : seul le secteur à deux directions de flux (1,1,0 et permutations) est réalisé".into())),
        }
    }

    pub fn controler_niveaux(&self, cotes: &[usize; 4], bloc: &ParamBloc) -> Resultat<()> {
        controler_pas(cotes[self.d[0]], bloc)
    }
}

fn controler_pas(cote: usize, bloc: &ParamBloc) -> Resultat<()> {
    if let Some(&b) = bloc.niveaux.iter().find(|&&b| b >= usize::BITS as usize || !cote.is_multiple_of(1usize << b)) {
        return Err(Erreur::Config(format!("--secteur-flux : 2^{b} ne divise pas le côté de flux {cote}")));
    }
    Ok(())
}

/// Secteur et directions d'enroulement sur le réseau (axes croissants, hors temps).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecteurCentre {
    Deux(SecteurFlux),
    Trois([usize; 3]),
}

impl SecteurCentre {
    /// `texte` = triplet sur les axes spatiaux (ordre croissant) ou quadruplet sur les axes 0..3.
    pub fn analyser(texte: &str, cotes: &[usize; 4], temps: usize) -> Resultat<SecteurCentre> {
        let e: Vec<i64> = texte
            .split(',')
            .map(|s| s.trim().parse::<i64>().map_err(|_| Erreur::Config(format!("--secteur-flux : entiers attendus, reçu « {texte} »"))))
            .collect::<Resultat<_>>()?;
        if temps > 3 {
            return Err(Erreur::Config("--secteur-flux : axe du temps hors 0..3".into()));
        }
        let axes: Vec<usize> = match e.len() {
            3 => (0..4).filter(|&d| d != temps).collect(),
            4 => (0..4).collect(),
            _ => return Err(Erreur::Config("--secteur-flux : triplet (axes spatiaux) ou quadruplet (axes 0..3) attendu".into())),
        };
        if e.iter().any(|&x| x != 0 && x != 1) {
            return Err(Erreur::Config("--secteur-flux : charges binaires (0 ou 1) attendues".into()));
        }
        let d: Vec<usize> = axes.iter().zip(&e).filter(|(_, &x)| x == 1).map(|(&a, _)| a).collect();
        let f = match *d.as_slice() {
            [a, b] => Self::Deux(SecteurFlux { d: [a, b] }),
            [a, b, c] => Self::Trois([a, b, c]),
            _ => return Err(Erreur::Config("--secteur-flux : deux ou trois directions de flux attendues".into())),
        };
        f.controler_axes(cotes, temps)?;
        Ok(f)
    }

    pub fn directions(&self) -> &[usize] {
        match self {
            Self::Deux(f) => &f.d,
            Self::Trois(d) => d,
        }
    }

    fn controler_axes(&self, cotes: &[usize; 4], temps: usize) -> Resultat<()> {
        let d = self.directions();
        let axes_invalides = match *self {
            Self::Deux(SecteurFlux { d: [a, b] }) => a == b,
            Self::Trois([a, b, c]) => !(a < b && b < c),
        };
        if temps > 3 || d.iter().any(|&a| a > 3) || axes_invalides {
            return Err(Erreur::Config("--secteur-flux : axes dans 0..3 et directions de flux distinctes requises (croissantes pour 1,1,1)".into()));
        }
        if d.contains(&temps) {
            return Err(Erreur::Config("--secteur-flux : une direction de flux est l'axe du temps".into()));
        }
        let cote = cotes[d[0]];
        if cote == 0 || d.iter().any(|&a| cotes[a] != cote) {
            let longueurs: Vec<_> = d.iter().map(|&a| cotes[a]).collect();
            return Err(Erreur::Config(format!("--secteur-flux : côtés de flux égaux et non nuls requis pour fermer l'escalier (axes {d:?}, côtés {longueurs:?})")));
        }
        Ok(())
    }

    pub fn controler_niveaux(&self, cotes: &[usize; 4], temps: usize, bloc: &ParamBloc) -> Resultat<()> {
        self.controler_axes(cotes, temps)?;
        bloc.controler(cotes, temps)?;
        match self {
            Self::Deux(f) => f.controler_niveaux(cotes, bloc),
            Self::Trois([a, _, _]) => controler_pas(cotes[*a], bloc),
        }
    }

    pub(super) fn n_sommes(&self) -> usize {
        match self {
            Self::Deux(_) => 6,
            Self::Trois(_) => 1,
        }
    }

    fn noms_bruts(&self) -> &[&str] {
        match self {
            Self::Deux(_) => &NOMS_BRUTS,
            Self::Trois(_) => &["escalier_111"],
        }
    }

    fn noms_base(&self) -> &[&str] {
        match self {
            Self::Deux(_) => &NOMS_BASE,
            Self::Trois(_) => self.noms_bruts(),
        }
    }

    pub(super) fn n_base(&self) -> usize {
        self.noms_base().len()
    }
}

// La disposition historique en mémoire et les noms locaux sont indépendants du choix des deux axes.
const FORMAT_DEUX_ENROULEMENTS: SecteurCentre = SecteurCentre::Deux(SecteurFlux { d: [0, 1] });

fn tr2(q: &Q) -> f64 {
    q[0]
}

/// Sommes de tranche S_q(t) (q = a+, a−, b+, b−, T1, T2) d'un champ F64, pas de superlien `l`. Résultat [t][6].
pub fn sommes_cpu(g: &GeoA, u: &[Q], temps: usize, f: &SecteurFlux, l: usize) -> Vec<[f64; 6]> {
    let tt = g.d[temps];
    let [d1, d2] = f.d;
    let n = g.d[d1] / l;
    let li = l as i64;
    let mut out = vec![[0f64; 6]; tt];
    for s in 0..g.v {
        let t = (s / g.st[temps]) % tt;
        let (mut l1, mut l2) = ([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
        let (mut s1, mut s2) = (s, s);
        for _ in 0..n {
            l1 = qmul(l1, u[d1 * g.v + s1]);
            s1 = decaler(g, s1, d1, li);
            l2 = qmul(l2, u[d2 * g.v + s2]);
            s2 = decaler(g, s2, d2, li);
        }
        let (mut bp, mut bm) = ([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
        let (mut sp, mut sm) = (s, s);
        for _ in 0..n {
            bp = qmul(bp, u[d1 * g.v + sp]);
            sp = decaler(g, sp, d1, li);
            bp = qmul(bp, u[d2 * g.v + sp]);
            sp = decaler(g, sp, d2, li);
            bm = qmul(bm, u[d1 * g.v + sm]);
            sm = decaler(g, sm, d1, li);
            sm = decaler(g, sm, d2, -li);
            bm = qmul(bm, qdag(u[d2 * g.v + sm]));
        }
        let o = &mut out[t];
        o[0] += tr2(&qmul(l1, l2));
        o[1] += tr2(&qmul(l1, qdag(l2)));
        o[2] += tr2(&bp);
        o[3] += tr2(&bm);
        o[4] += tr2(&l1);
        o[5] += tr2(&l2);
    }
    out
}

fn sommes_111_cpu(g: &GeoA, u: &[Q], temps: usize, d: [usize; 3], l: usize) -> Vec<f64> {
    let tt = g.d[temps];
    let n = g.d[d[0]] / l;
    let abc = [1, 2, 3].repeat(n);
    let acb = [1, 3, 2].repeat(n);
    let mut out = vec![0.0; tt];
    for s in 0..g.v {
        let t = (s / g.st[temps]) % tt;
        let p = trace_chemin_l(g, u, &d, &abc, s, l);
        let q = trace_chemin_l(g, u, &d, &acb, s, l);
        out[t] += 0.5 * (p + q);
    }
    out
}

/// Niveaux de la mesure (APE cumulatifs `niveaux`, puis blocage) : [(champ, pas de superlien)] dans l'ordre des niveaux.
pub fn champs_niveaux(g: &GeoA, u: &[Q], temps: usize, alpha: f64, niveaux: &[usize], bloc: &ParamBloc) -> Vec<(Vec<Q>, usize)> {
    let mut v = Vec::new();
    let mut cur = u.to_vec();
    let mut fait = 0;
    for &n in niveaux {
        cur = lisser(g, &cur, temps, alpha, n - fait);
        fait = n;
        v.push((cur.clone(), 1usize));
    }
    for (b, c) in champs_bloques(g, u, temps, alpha, bloc) {
        v.push((c, 1usize << b));
    }
    v
}

/// Sommes de tranche [niveau][t][q] sur tous les niveaux.
fn sommes_niveaux_centre_cpu(g: &GeoA, u: &[Q], temps: usize, f: &SecteurCentre, alpha: f64, niveaux: &[usize], bloc: &ParamBloc) -> Vec<f64> {
    let mut o = Vec::new();
    for (c, l) in champs_niveaux(g, u, temps, alpha, niveaux, bloc) {
        match f {
            SecteurCentre::Deux(f) => o.extend(sommes_cpu(g, &c, temps, f, l).into_iter().flatten()),
            SecteurCentre::Trois(d) => o.extend(sommes_111_cpu(g, &c, temps, *d, l)),
        }
    }
    o
}

/// Sommes de tranche du secteur à deux enroulements, dans l'ordre [niveau][t][6].
pub fn sommes_niveaux_cpu(g: &GeoA, u: &[Q], temps: usize, f: &SecteurFlux, alpha: f64, niveaux: &[usize], bloc: &ParamBloc) -> Vec<f64> {
    sommes_niveaux_centre_cpu(g, u, temps, &SecteurCentre::Deux(*f), alpha, niveaux, bloc)
}

/// Référence CPU F64 sur un champ donné, isotrope ou anisotrope, sur un tore périodique à coutures droites.
/// `xi0` règle la génération du champ, les chemins et le lissage spatial n'en dépendent pas.
/// Sortie [t][niveau][opérateur], comme `ops_flux.bin`.
pub fn mesurer_cpu(g: &GeoA, u: &[Q], temps: usize, f: &SecteurCentre, alpha: f64, niveaux: &[usize], bloc: &ParamBloc) -> Resultat<Vec<f64>> {
    if g.couture.is_some() {
        return Err(Erreur::Config("--secteur-flux : le chemin requiert des coutures périodiques sans décalage".into()));
    }
    f.controler_niveaux(&g.d, temps, bloc)?;
    if u.len() != 4 * g.v || niveaux.is_empty() || niveaux.windows(2).any(|w| w[0] >= w[1]) {
        return Err(Erreur::Config("--secteur-flux : champ de 4V liens et niveaux APE croissants non vides requis".into()));
    }
    let sommes = sommes_niveaux_centre_cpu(g, u, temps, f, alpha, niveaux, bloc);
    Ok(assembler_centre(f, &sommes, niveaux.len() + bloc.niveaux.len(), g.d[temps]))
}

/// Assemblage hôte : sommes [niveau][t][q] → opérateurs bruts [t][niveau][k].
pub(super) fn assembler_centre(f: &SecteurCentre, sommes: &[f64], nlev: usize, tt: usize) -> Vec<f64> {
    let nq = f.n_sommes();
    let nb = f.noms_bruts().len();
    let mut o = vec![0f64; tt * nlev * nb];
    for lev in 0..nlev {
        for t in 0..tt {
            let s = &sommes[(lev * tt + t) * nq..(lev * tt + t + 1) * nq];
            let d = &mut o[(t * nlev + lev) * nb..(t * nlev + lev + 1) * nb];
            match f {
                SecteurCentre::Deux(_) => d.copy_from_slice(&[s[0], s[1], s[2], s[3], s[4] * s[5], s[4], s[5]]),
                SecteurCentre::Trois(_) => d.copy_from_slice(s),
            }
        }
    }
    o
}

/// Assemblage du secteur à deux enroulements, [niveau][t][6] → [t][niveau][7].
pub fn assembler(sommes: &[f64], nlev: usize, tt: usize) -> Vec<f64> {
    assembler_centre(&FORMAT_DEUX_ENROULEMENTS, sommes, nlev, tt)
}

/// Base de corrélation [(niveau,k)][t] à partir des bruts, dans l'ordre du manifeste.
pub fn base_corr_centre(f: &SecteurCentre, ops: &[f64], nlev: usize, tt: usize) -> Vec<Vec<[f64; 2]>> {
    let nb = f.noms_bruts().len();
    let nc = f.n_base();
    let mut v = vec![vec![[0f64; 2]; tt]; nlev * nc];
    for t in 0..tt {
        for lev in 0..nlev {
            let r = &ops[(t * nlev + lev) * nb..(t * nlev + lev + 1) * nb];
            match f {
                SecteurCentre::Deux(_) => {
                    let b = [r[0] + r[1], r[2] + r[3], r[4], r[5], r[6]];
                    for k in 0..NOPS_BASE {
                        v[lev * nc + k][t][0] = b[k];
                    }
                }
                SecteurCentre::Trois(_) => v[lev][t][0] = r[0],
            }
        }
    }
    v
}

/// Base de corrélation du secteur à deux enroulements, dans l'ordre de `NOMS_BASE`.
pub fn base_corr(ops: &[f64], nlev: usize, tt: usize) -> Vec<Vec<[f64; 2]>> {
    base_corr_centre(&FORMAT_DEUX_ENROULEMENTS, ops, nlev, tt)
}

/// C_ab(τ) symétrisé sur le temps périodique (même fonction que `corr.csv`) pour la base de corrélation.
pub fn corr_flux_centre(f: &SecteurCentre, ops: &[f64], nlev: usize, tt: usize, tau_max: usize) -> Vec<Vec<f64>> {
    super::correlateur(&base_corr_centre(f, ops, nlev, tt), tau_max, Mutant::Aucun)
}

/// Corrélateur temporel du secteur à deux enroulements.
pub fn corr_flux(ops: &[f64], nlev: usize, tt: usize, tau_max: usize) -> Vec<Vec<f64>> {
    corr_flux_centre(&FORMAT_DEUX_ENROULEMENTS, ops, nlev, tt, tau_max)
}

/// ⟨O_a⟩_t des opérateurs bruts, dans l'ordre du manifeste (jamais soustrait).
pub fn moyennes_centre(f: &SecteurCentre, ops: &[f64], nlev: usize, tt: usize) -> Vec<f64> {
    let m = nlev * f.noms_bruts().len();
    (0..m).map(|i| (0..tt).map(|t| ops[t * m + i]).sum::<f64>() / tt as f64).collect()
}

/// Moyennes temporelles brutes du secteur à deux enroulements.
pub fn moyennes(ops: &[f64], nlev: usize, tt: usize) -> Vec<f64> {
    moyennes_centre(&FORMAT_DEUX_ENROULEMENTS, ops, nlev, tt)
}

fn ordre_operateurs(f: &SecteurCentre, niv_man: &serde_json::Value) -> (Vec<String>, Vec<String>) {
    let noms: Vec<String> = niv_man.as_array().map(|a| a.iter().map(|x| x.as_str().map(String::from).unwrap_or_else(|| x.to_string())).collect()).unwrap_or_default();
    let ops = noms.iter().flat_map(|n| f.noms_bruts().iter().map(move |o| format!("{n}:{o}"))).collect();
    let base = noms.iter().flat_map(|n| f.noms_base().iter().map(move |o| format!("{n}:{o}"))).collect();
    (ops, base)
}

/// Entrée « secteur_flux » du manifeste pour le secteur à deux enroulements.
pub fn manifeste(f: &SecteurFlux, cotes: &[usize; 4], temps: usize, niv_man: &serde_json::Value, tau_max: usize) -> serde_json::Value {
    let (ops, base) = ordre_operateurs(&FORMAT_DEUX_ENROULEMENTS, niv_man);
    serde_json::json!({
        "secteur": "(1,1,0) : deux directions de flux", "directions_flux": f.d, "temps": temps, "cote_flux": cotes[f.d[0]], "cotes": cotes,
        "impulsion_totale": "nulle (somme sur tous les points de la tranche de temps)",
        "fichiers": {
            "ops_flux.bin": "f64 little-endian, [configuration][t][operateur], operateur = i_niveau*7 + k ; sommes de tranche (pas de moyenne) ; c = (somme T1)*(somme T2)",
            "corr_flux.csv": "cfg,px,py (0,0), puis c<ia>_<ib>_<tau> : C_ab(tau) symetrise sur le temps periodique pour la base de correlation (indice = i_niveau*5 + k), tau <= tau_max",
            "flux_moy.csv": "cfg,operateur,moyenne_t : <O_a>_t des operateurs bruts (nul en moyenne, symetrie de centre)",
        },
        "ordre_operateurs_bruts": ops, "ordre_base_correlation": base, "niveaux": niv_man, "tau_max": tau_max,
        "definition": "a+- = 1/2 Tr[L1 L2^(+-)], L_i = ligne non tracee autour de la direction i ; b+- = escalier (+1,+-2) repete L/l fois ; T_i = 1/2 Tr L_i ; base : a++a-, b++b-, c, T1, T2 (parite sous x2 -> -x2)",
    })
}

/// Entrée « secteur_flux » du manifeste pour le secteur mesuré.
pub(super) fn manifeste_centre(f: &SecteurCentre, p: &ParamTorelon, niv_man: &serde_json::Value) -> serde_json::Value {
    let (cotes, temps, tau_max) = (p.cotes, p.temps, p.tau_max);
    let [a, b, c] = match f {
        SecteurCentre::Deux(f) => return manifeste(f, &cotes, temps, niv_man, tau_max),
        SecteurCentre::Trois(d) => *d,
    };
    let (ops, base) = ordre_operateurs(f, niv_man);
    let tt = cotes[temps];
    let cote = cotes[a];
    let mut pas = vec![1; p.niveaux.len()];
    pas.extend(p.bloc.niveaux.iter().map(|&b| 1usize << b));
    let repetitions: Vec<_> = pas.iter().map(|l| cote / l).collect();
    let mut charge = [1; 4];
    charge[temps] = 0;
    serde_json::json!({
        "secteur": "(1,1,1) : trois directions de flux",
        "charge_centre_axes_spatiaux": [1, 1, 1], "charge_centre_axes_reseau": charge,
        "directions_flux": [a, b, c], "temps": temps, "cote_flux": cote, "cotes": cotes,
        "xi0_action": p.xi0.unwrap_or(1.0),
        "geometrie": "tore périodique à coutures sans décalage",
        "unites": {"espace": "pas spatial a_s du réseau fin", "temps": "pas temporel a_t du réseau fin (t et tau)", "operateur": "sans dimension"},
        "impulsion_totale": "nulle (somme sur tous les points de la tranche de temps)",
        "fichiers": {
            "ops_flux.bin": "f64 little-endian sans en-tête, [configuration][t][operateur], operateur = i_niveau",
            "corr_flux.csv": "cfg,px,py (0,0), puis c<ia>_<ib>_<tau> pour i_a <= i_b et 0 <= tau <= tau_max, moyenne symétrisée sur le temps périodique sans soustraction",
            "flux_moy.csv": "cfg,operateur,moyenne_t : <O>_t sans soustraction",
        },
        "dimensions_ops_par_configuration": [tt, ops.len()], "octets_par_configuration": tt * ops.len() * 8,
        "ordre_operateurs_bruts": ops, "ordre_base_correlation": base, "niveaux": niv_man,
        "longueurs_superlien": pas, "repetitions_par_chemin": repetitions, "tau_max": tau_max,
        "motifs_axes_positifs": [[a, b, c], [a, c, b]],
        "definition": "Par site, 1/2 de la somme des deux demi-traces SU(2) des produits ordonnés de superliens, chaque motif répété L/ℓ fois. Somme sur tous les sites de la tranche du réseau fin, sans division par L^3. Déplacement L selon chacun des trois axes, zéro en temps. Un motif représentatif par classe de permutation cyclique, sans moyenne sur les huit choix de signe.",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commun::Precision;
    use crate::su2::torsion::Torsion;
    use crate::su2::tube::reseau::{pile_aniso, ChaineTube, Pile};

    fn config(d: [usize; 4], beta: f64, graine: u64, n: usize) -> (GeoA, Vec<Q>) {
        let g = GeoA::nouvelle(d).unwrap();
        let pile = Pile::depuis_torsion(&g, &Torsion::analyser("aucune").unwrap());
        let mut c = ChaineTube::nouvelle(g.clone(), graine);
        c.init(true);
        for _ in 0..n {
            c.pas(beta, 2, &pile);
        }
        (g, c.u)
    }
    fn fl(d: [usize; 2]) -> SecteurFlux {
        SecteurFlux { d }
    }
    const D: [usize; 4] = [4, 4, 4, 4];
    fn ecart(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).abs()))
    }
    fn plat(v: &[[f64; 6]]) -> Vec<f64> {
        v.iter().flatten().copied().collect()
    }

    #[test]
    fn analyse_et_refus() {
        let c = [8usize, 8, 8, 32];
        assert_eq!(SecteurCentre::analyser("1,1,0", &c, 3).unwrap().directions(), &[0, 1]);
        assert_eq!(SecteurCentre::analyser("0,1,1", &c, 3).unwrap().directions(), &[1, 2]);
        assert_eq!(SecteurCentre::analyser("1,1,0,0", &c, 3).unwrap().directions(), &[0, 1]);
        assert!(SecteurCentre::analyser("1,1,0,1", &c, 3).is_err(), "axe du temps");
        assert_eq!(SecteurCentre::analyser("1,1,1", &c, 3).unwrap(), SecteurCentre::Trois([0, 1, 2]));
        assert!(SecteurCentre::analyser("1,0,0", &c, 3).is_err());
        assert!(SecteurCentre::analyser("1,2,0", &c, 3).is_err());
        assert!(SecteurCentre::analyser("1,1", &c, 3).is_err());
        assert!(SecteurCentre::analyser("1,1,0", &[8, 6, 8, 32], 3).is_err(), "côtés différents");
        assert!(SecteurCentre::analyser("a,1,0", &c, 3).is_err());
        // temps = 0 : triplet sur les axes 1,2,3
        assert_eq!(SecteurCentre::analyser("1,0,1", &[32, 8, 8, 8], 0).unwrap().directions(), &[1, 3]);
    }

    #[test]
    fn liens_unites_valeurs_exactes() {
        let g = GeoA::nouvelle([4, 4, 4, 6]).unwrap();
        let u = vec![[1.0, 0.0, 0.0, 0.0]; 4 * g.v];
        let v3 = 64.0;
        for l in [1usize, 2] {
            for t in sommes_cpu(&g, &u, 3, &fl([0, 1]), l) {
                assert_eq!(t, [v3; 6], "pas {l}");
            }
        }
        let ops = assembler(&sommes_niveaux_cpu(&g, &u, 3, &fl([0, 1]), 0.5, &[0, 2], &ParamBloc::vide()), 2, 6);
        for t in 0..6 {
            for lev in 0..2 {
                assert_eq!(&ops[(t * 2 + lev) * 7..(t * 2 + lev + 1) * 7], &[v3, v3, v3, v3, v3 * v3, v3, v3]);
            }
        }
    }

    #[test]
    fn invariance_de_jauge() {
        let (g, u) = config(D, 2.3, 5, 6);
        let mut u2 = u.clone();
        crate::su2::spectro::transformer_jauge(&g, &mut u2, 77);
        assert!(u.iter().zip(&u2).any(|(a, b)| a != b));
        let bl = ParamBloc { niveaux: vec![1], ape_bloc: 0, lissage_bloc: 1, p_bloc: 1.0 };
        // le blocage n'est pas covariant à la jauge exactement au même sens que l'APE pur : niveaux APE seuls pour 1e-12, blocs aussi car covariants
        let a = sommes_niveaux_cpu(&g, &u, 3, &fl([0, 1]), 0.5, &[0, 2], &bl);
        let b = sommes_niveaux_cpu(&g, &u2, 3, &fl([0, 1]), 0.5, &[0, 2], &bl);
        let e = ecart(&a, &b);
        eprintln!("invariance de jauge flux : écart max {e:.3e} (échelle {:.3e})", a.iter().fold(0f64, |m, x| m.max(x.abs())));
        assert!(e < 1e-12);
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
    fn centre_signes() {
        let (g, mut u) = config(D, 2.3, 6, 6);
        let a = sommes_cpu(&g, &u, 3, &fl([0, 1]), 1);
        for s in 0..g.v {
            if g.coords(s)[0] == 0 {
                for k in 0..4 {
                    u[s][k] = -u[s][k]; // direction 0 (indice 0·v + s)
                }
            }
        }
        let b = sommes_cpu(&g, &u, 3, &fl([0, 1]), 1);
        for t in 0..4 {
            for q in 0..4 {
                assert!((b[t][q] + a[t][q]).abs() < 1e-12, "q={q}");
            }
            assert!((b[t][4] + a[t][4]).abs() < 1e-12, "T1");
            assert!((b[t][5] - a[t][5]).abs() < 1e-12, "T2 inchangé");
        }
        assert!(a.iter().any(|r| r[0].abs() > 1e-3));
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
    fn translation_et_echange() {
        let (g, u) = config(D, 2.3, 7, 6);
        let a = sommes_cpu(&g, &u, 3, &fl([0, 1]), 1);
        // translation d'un site le long de l'axe 2 (somme de tranche : inchangée) et de l'axe 3 (temps : t → t+1)
        for ax in [2usize, 3] {
            let mut u2 = u.clone();
            for mu in 0..4 {
                for s in 0..g.v {
                    let mut x = g.coords(s);
                    x[ax] = (x[ax] + 1) % D[ax];
                    u2[mu * g.v + g.site(&x)] = u[mu * g.v + s];
                }
            }
            let b = sommes_cpu(&g, &u2, 3, &fl([0, 1]), 1);
            for t in 0..4 {
                let tb = if ax == 3 { (t + 1) % 4 } else { t };
                assert!(ecart(&a[t], &b[tb]) < 1e-12, "axe {ax} t {t}");
            }
        }
        // échange 1 ↔ 2 : a+, a−, T1↔T2 inchangés (trace cyclique)
        let e = sommes_cpu(&g, &u, 3, &fl([1, 0]), 1);
        for t in 0..4 {
            assert!((a[t][0] - e[t][0]).abs() < 1e-12 && (a[t][1] - e[t][1]).abs() < 1e-12);
            assert!((a[t][4] - e[t][5]).abs() < 1e-12 && (a[t][5] - e[t][4]).abs() < 1e-12);
        }
        // un autre couple de directions (1,2) donne autre chose que (0,1) : l'option lit bien la direction demandée
        let h = sommes_cpu(&g, &u, 3, &fl([1, 2]), 1);
        assert!(ecart(&plat(&a), &plat(&h)) > 1e-6);
    }

    #[test]
    fn escalier_a_partir_de_chemins_explicites() {
        // b+ et b− contre un calcul indépendant par chemins explicites (pas à pas), pas ℓ = 1 et 2
        let (g, u) = config([4, 4, 4, 4], 2.3, 8, 6);
        for l in [1usize, 2] {
            let s = sommes_cpu(&g, &u, 3, &fl([0, 2]), l);
            let mut bp = vec![0f64; 4];
            let mut bm = vec![0f64; 4];
            for s0 in 0..g.v {
                let t = g.coords(s0)[3];
                for (sig, acc) in [(1i64, &mut bp), (-1i64, &mut bm)] {
                    let mut x: Vec<i64> = g.coords(s0).iter().map(|&c| c as i64).collect();
                    let mut p: Q = [1.0, 0.0, 0.0, 0.0];
                    for _ in 0..4 / l {
                        let idx = |x: &Vec<i64>| g.site(&[x[0].rem_euclid(4) as usize, x[1].rem_euclid(4) as usize, x[2].rem_euclid(4) as usize, x[3].rem_euclid(4) as usize]);
                        p = qmul(p, u[idx(&x)]);
                        x[0] += l as i64;
                        if sig == 1 {
                            p = qmul(p, u[2 * g.v + idx(&x)]);
                            x[2] += l as i64;
                        } else {
                            x[2] -= l as i64;
                            p = qmul(p, qdag(u[2 * g.v + idx(&x)]));
                        }
                    }
                    acc[t] += p[0];
                }
            }
            for t in 0..4 {
                assert!((s[t][2] - bp[t]).abs() < 1e-12 && (s[t][3] - bm[t]).abs() < 1e-12, "l={l} t={t}");
            }
        }
    }

    fn bloc_111() -> ParamBloc {
        ParamBloc { niveaux: vec![1], ape_bloc: 1, lissage_bloc: 1, p_bloc: 1.0 }
    }

    fn params_111(temps: usize, precision: Precision) -> ParamTorelon {
        let mut cotes = [4; 4];
        cotes[temps] = 6;
        ParamTorelon {
            cotes,
            w: (temps + 1) % 4,
            temps,
            niveaux: vec![0, 2],
            alpha: 0.5,
            moments: vec![[0, 0]],
            tau_max: 3,
            precision,
            nor: 2,
            betas: vec![2.39; 2],
            graines: vec![7, 8],
            bloc: bloc_111(),
            xi0: Some(1.4),
            ops_bruts: false,
            profil: vec![],
            profil_niveau: None,
        }
    }

    #[test]
    fn flux111_analyse_geometrie_et_manifeste() {
        for temps in 0..4 {
            let p = params_111(temps, Precision::F64);
            let axes: Vec<_> = (0..4).filter(|&a| a != temps).collect();
            let quad: Vec<_> = (0..4).map(|a| if a == temps { "0" } else { "1" }).collect();
            let f = SecteurCentre::analyser("1,1,1", &p.cotes, temps).unwrap();
            assert_eq!(f.directions(), axes);
            assert_eq!(f, SecteurCentre::analyser(&quad.join(","), &p.cotes, temps).unwrap());
            assert!(SecteurCentre::analyser("1,1,1,1", &p.cotes, temps).is_err());
            let mut cotes = p.cotes;
            cotes[axes[2]] = 6;
            assert!(SecteurCentre::analyser("1,1,1", &cotes, temps).is_err());
            let mut bloc = bloc_111();
            bloc.niveaux = vec![2];
            assert!(f.controler_niveaux(&[6, 6, 6, 6], temps, &bloc).is_err()); // 4 ne divise pas 6
            bloc.niveaux = vec![usize::BITS as usize];
            assert!(f.controler_niveaux(&p.cotes, temps, &bloc).is_err());

            let niv = serde_json::json!([0, 2, "B1s1"]);
            let m = manifeste_centre(&f, &p, &niv);
            assert_eq!(m["directions_flux"], serde_json::json!(axes));
            assert_eq!(m["temps"], serde_json::json!(temps));
            assert_eq!(m["xi0_action"], serde_json::json!(1.4));
            assert_eq!(m["charge_centre_axes_reseau"][temps], serde_json::json!(0));
            assert_eq!(m["charge_centre_axes_spatiaux"], serde_json::json!([1, 1, 1]));
            assert_eq!(m["longueurs_superlien"], serde_json::json!([1, 1, 2]));
            assert_eq!(m["repetitions_par_chemin"], serde_json::json!([4, 4, 2]));
            assert_eq!(m["dimensions_ops_par_configuration"], serde_json::json!([6, 3]));
            assert_eq!(m["octets_par_configuration"], serde_json::json!(144));
            assert_eq!(m["ordre_operateurs_bruts"], serde_json::json!(["0:escalier_111", "2:escalier_111", "B1s1:escalier_111"]));
            assert_eq!(m["ordre_operateurs_bruts"], m["ordre_base_correlation"]);
        }
        let g = GeoA::avec_couture([4, 4, 4, 6], 0, [0, 1, 0, 0]).unwrap();
        let f = SecteurCentre::analyser("1,1,1", &g.d, 3).unwrap();
        let unite = vec![[1.0, 0.0, 0.0, 0.0]; 4 * g.v];
        assert!(matches!(mesurer_cpu(&g, &unite, 3, &f, 0.5, &[0], &bloc_111()), Err(Erreur::Config(m)) if m.contains("coutures")));
    }

    #[test]
    fn flux111_unite_et_dimensions() {
        for temps in 0..4 {
            let p = params_111(temps, Precision::F64);
            let g = GeoA::nouvelle(p.cotes).unwrap();
            let f = SecteurCentre::analyser("1,1,1", &g.d, temps).unwrap();
            let u = vec![[1.0, 0.0, 0.0, 0.0]; 4 * g.v];
            let (tt, nlev, vol) = (6, 3, 64.0);
            let ops = mesurer_cpu(&g, &u, temps, &f, p.alpha, &p.niveaux, &p.bloc).unwrap();
            assert_eq!(ops, vec![vol; tt * nlev]);
            assert_eq!(moyennes_centre(&f, &ops, nlev, tt), vec![vol; 3]);
            assert_eq!(corr_flux_centre(&f, &ops, nlev, tt, p.tau_max), vec![vec![vol * vol; 4]; 6]);
        }
        let f = SecteurCentre::Trois([0, 1, 2]);
        let ops = assembler_centre(&f, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        assert_eq!(ops, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(moyennes_centre(&f, &ops, 2, 3), [2.0, 5.0]);
        assert_eq!(corr_flux_centre(&f, &ops, 2, 3, 1)[1], [32.0 / 3.0, 29.0 / 3.0]);
    }

    /// Champ de test dyadique, unitaire et identique au bit en F32/F64 sur HIP et CUDA.
    fn champ_111(g: &GeoA) -> Vec<Q> {
        use crate::heritage::su2_hb::philox4x32_10;
        (0..4 * g.v)
            .map(|i| {
                let n = philox4x32_10([i as u32, 0, 0, 0], 17, 29)[0];
                if n & 1 == 0 {
                    [1.0, 0.0, 0.0, 0.0]
                } else {
                    [2, 4, 8, 16].map(|b| if n & b == 0 { 0.5 } else { -0.5 })
                }
            })
            .collect()
    }

    fn centre(g: &GeoA, u: &[Q], axe: usize) -> Vec<Q> {
        let mut v = u.to_vec();
        for s in 0..g.v {
            if g.coords(s)[axe] == g.d[axe] - 1 {
                v[axe * g.v + s] = u[axe * g.v + s].map(|q| -q);
            }
        }
        v
    }

    #[test]
    fn flux111_fermeture_et_chemins_explicites() {
        let g = GeoA::nouvelle([4, 6, 4, 4]).unwrap();
        let u = champ_111(&g);
        for (c, l) in [(u.clone(), 1), (champs_bloques(&g, &u, 1, 0.5, &bloc_111()).pop().unwrap().1, 2)] {
            let mut ref_cpu = vec![0.0; 6];
            for s in 0..g.v {
                let x0 = g.coords(s);
                for motif in [[0, 2, 3], [0, 3, 2]] {
                    let mut x = x0;
                    let mut q = [1.0, 0.0, 0.0, 0.0];
                    for d in motif.into_iter().cycle().take(3 * 4 / l) {
                        let site = g.site(&std::array::from_fn(|a| x[a] % g.d[a]));
                        q = qmul(q, c[d * g.v + site]);
                        x[d] += l;
                    }
                    assert_eq!(x, [x0[0] + 4, x0[1], x0[2] + 4, x0[3] + 4]);
                    ref_cpu[x0[1]] += 0.5 * q[0];
                }
            }
            assert!(ecart(&ref_cpu, &sommes_111_cpu(&g, &c, 1, [0, 2, 3], l)) < 1e-12);
        }
    }

    #[test]
    fn flux111_jauge_centre_permutations_sur_champ_anisotrope() {
        let p = params_111(2, Precision::F64);
        let temps = p.temps;
        let g = GeoA::nouvelle(p.cotes).unwrap();
        let f = SecteurCentre::analyser("1,1,1", &g.d, temps).unwrap();
        let pile = pile_aniso(Pile::depuis_torsion(&g, &Torsion::triviale()), &g, p.xi0.unwrap(), temps);
        let mut chaine = ChaineTube::nouvelle(g.clone(), 27);
        chaine.init(true);
        for _ in 0..6 {
            chaine.pas(2.39, 2, &pile);
        }
        let u = &chaine.u;
        let mesurer = |v: &[Q]| mesurer_cpu(&g, v, temps, &f, p.alpha, &p.niveaux, &p.bloc).unwrap();
        let a = mesurer(u);
        for niv in 0..3 {
            assert!(a.iter().skip(niv).step_by(3).any(|o| o.abs() > 1e-3));
        }
        let mut v = u.clone();
        super::super::transformer_jauge(&g, &mut v, 29);
        assert!(ecart(&a, &mesurer(&v)) < 1e-10);
        let neg: Vec<_> = a.iter().map(|v| -v).collect();
        for &axe in f.directions() {
            assert!(ecart(&neg, &mesurer(&centre(&g, u, axe))) < 1e-10, "axe {axe}");
        }
        assert!(ecart(&a, &mesurer(&centre(&g, u, temps))) < 1e-10);

        // Action de S3 sur les coordonnées et les directions des liens ; la direction du temps reste 2.
        for perm in [[0, 1, 3], [0, 3, 1], [1, 0, 3], [1, 3, 0], [3, 0, 1], [3, 1, 0]] {
            let axes = [perm[0], perm[1], temps, perm[2]];
            let mut v = u.clone();
            for s in 0..g.v {
                let x = g.coords(s);
                let y = axes.map(|d| x[d]);
                let dest = g.site(&y);
                for (a, &d) in axes.iter().enumerate() {
                    v[a * g.v + dest] = u[d * g.v + s];
                }
            }
            assert!(ecart(&a, &mesurer(&v)) < 1e-10, "permutation {perm:?}");
        }
    }

    /// carte contre processeur : `#[ignore]`, exige une carte HIP ou CUDA compatible.
    #[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
    fn carte_contre_cpu(backend: &str, prec: crate::commun::Precision, tol: f64) {
        use crate::su2::spectro::gpu_torelon::{ParamTorelon, TorelonGpu};
        let d = [8usize, 8, 8, 8];
        let app = crate::su2::cli::ouvrir_backend(backend).unwrap().expect("GPU requis");
        let bl = ParamBloc { niveaux: vec![1], ape_bloc: 1, lissage_bloc: 1, p_bloc: 1.0 };
        let p = ParamTorelon {
            cotes: d,
            w: 2,
            temps: 3,
            niveaux: vec![0, 2],
            alpha: 0.5,
            moments: vec![[0, 0]],
            tau_max: 3,
            precision: prec,
            nor: 4,
            betas: vec![2.39, 2.39],
            graines: vec![7, 8],
            bloc: bl.clone(),
            xi0: None,
            ops_bruts: false,
            profil: vec![],
            profil_niveau: None,
        };
        let mut tg = TorelonGpu::nouveau(&app, &p).unwrap();
        tg.init().unwrap();
        tg.avancer(40).unwrap();
        let f = fl([0, 1]);
        let res = tg.mesurer_flux(&f).unwrap();
        let g = GeoA::nouvelle(d).unwrap();
        for c in 0..2 {
            let raw = tg.boite.telecharger(c).unwrap();
            let u: Vec<Q> = raw.chunks(4).map(|x| [x[0], x[1], x[2], x[3]]).collect();
            let r = assembler(&sommes_niveaux_cpu(&g, &u, 3, &f, 0.5, &[0, 2], &bl), 3, 8);
            let sc = r.iter().fold(0f64, |m, x| m.max(x.abs()));
            let e = ecart(&r, &res[c]) / sc;
            eprintln!("{backend} {prec:?} chaîne {c} : écart flux carte/processeur {e:.3e} (échelle {sc:.3e})");
            assert!(e < tol);
        }
    }
    #[test]
    #[ignore]
    fn flux_carte_f64_hip() {
        carte_contre_cpu("hip", crate::commun::Precision::F64, 1e-10);
    }
    #[test]
    #[ignore]
    fn flux_carte_f32_hip() {
        carte_contre_cpu("hip", crate::commun::Precision::F32, 1e-5);
    }
    #[test]
    #[ignore]
    fn flux_carte_f64_cuda() {
        carte_contre_cpu("cuda", crate::commun::Precision::F64, 1e-10);
    }
    #[test]
    #[ignore]
    fn flux_carte_f32_cuda() {
        carte_contre_cpu("cuda", crate::commun::Precision::F32, 1e-5);
    }

    fn carte_111_contre_cpu(backend: &str, precision: Precision, tol: f64) {
        use crate::su2::spectro::gpu_torelon::TorelonGpu;
        let app = crate::su2::cli::ouvrir_backend(backend).unwrap().expect("GPU requis");
        for temps in 0..4 {
            let mut p = params_111(temps, precision);
            p.cotes = [8; 4];
            p.cotes[temps] = 16;
            let g = GeoA::nouvelle(p.cotes).unwrap();
            let f = SecteurCentre::analyser("1,1,1", &g.d, temps).unwrap();
            let mut tg = TorelonGpu::nouveau(&app, &p).unwrap();
            let champ = champ_111(&g);
            let unite = vec![[1.0, 0.0, 0.0, 0.0]; 4 * g.v];
            tg.boite.televerser(0, champ.as_flattened()).unwrap();
            tg.boite.televerser(1, unite.as_flattened()).unwrap();

            let cpu = mesurer_cpu(&g, &champ, temps, &f, p.alpha, &p.niveaux, &p.bloc).unwrap();
            let res = tg.mesurer_flux_centre(&f).unwrap();
            assert_eq!(res[0].len(), cpu.len());
            assert_eq!(res[1], vec![512.0; cpu.len()]);
            let sc = cpu.iter().fold(1.0f64, |m, v| m.max(v.abs()));
            let e = ecart(&cpu, &res[0]) / sc;
            assert!(e < tol, "{backend} {precision:?}, temps={temps}, écart relatif {e:.3e}");

            let mut chaine = ChaineTube::nouvelle(g.clone(), 7);
            chaine.u = champ.clone();
            let (ss, st) = chaine.plaquettes_ss_st(temps);
            let avant = tg.mesurer().unwrap();
            let (ss_gpu, st_gpu) = avant[0].plaq_st.unwrap();
            assert!((ss - st).abs() > 1e-6, "fixture doit distinguer les deux classes de plaquettes");
            assert!((ss - ss_gpu).abs() < tol && (st - st_gpu).abs() < tol);

            // Le chemin historique et son tampon peuvent servir entre deux mesures 111 sur le même champ.
            let mut axes = f.directions().iter().copied();
            let f110 = SecteurCentre::Deux(fl([axes.next().unwrap(), axes.next().unwrap()]));
            let flux110 = tg.mesurer_flux_centre(&f110).unwrap();
            let ref110 = mesurer_cpu(&g, &champ, temps, &f110, p.alpha, &p.niveaux, &p.bloc).unwrap();
            let sc110 = ref110.iter().fold(1.0f64, |m, v| m.max(v.abs()));
            assert!(ecart(&ref110, &flux110[0]) / sc110 < tol);
            let apres = tg.mesurer().unwrap();
            for (a, b) in avant.iter().zip(&apres) {
                assert_eq!(a.corr, b.corr);
                assert_eq!(a.phi_moyen, b.phi_moyen);
                assert_eq!(a.plaq_st, b.plaq_st);
                assert_eq!(a.plaq, b.plaq);
                assert_eq!(a.poly_w, b.poly_w);
            }

            let t0 = std::time::Instant::now();
            for _ in 0..16 {
                assert_eq!(tg.mesurer_flux_centre(&f).unwrap(), res);
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / 16.0;
            eprintln!("{backend} {precision:?} {:?}, temps={temps}, xi0={}: écart relatif max {e:.3e}, {ms:.3} ms/lot (lissage, noyaux et remontée de 2 chaînes)", g.d, p.xi0.unwrap());

            let neg: Vec<_> = res[0].iter().map(|x| -x).collect();
            for &axe in f.directions() {
                tg.boite.televerser(0, centre(&g, &champ, axe).as_flattened()).unwrap();
                let change = tg.mesurer_flux_centre(&f).unwrap();
                assert!(ecart(&neg, &change[0]) / sc < tol, "centre axe {axe}, temps={temps}");
            }
        }
    }

    #[test]
    #[ignore]
    fn flux111_carte_f64_hip() {
        carte_111_contre_cpu("hip", Precision::F64, 1e-10);
    }

    #[test]
    #[ignore]
    fn flux111_carte_f32_hip() {
        carte_111_contre_cpu("hip", Precision::F32, 1e-5);
    }

    #[test]
    #[ignore]
    fn flux111_carte_f64_cuda() {
        carte_111_contre_cpu("cuda", Precision::F64, 1e-10);
    }

    #[test]
    #[ignore]
    fn flux111_carte_f32_cuda() {
        carte_111_contre_cpu("cuda", Precision::F32, 1e-5);
    }
}
