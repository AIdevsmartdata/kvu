//! Moteur GPU F64 de la campagne « une boîte » (branche `campagne/une-boite`) : boîtes RECTANGULAIRES (côtés pairs quelconques),
//! torsion de 't Hooft dans un ou plusieurs plans (pile de coin (ℓ_μ − 1, ℓ_ν − 1), convention de `docs/TORSION.md`), poids RÉEL `s`
//! de la feuille (s = −1 : torsion ; s = +1 : témoin périodique de même masque de mesure), et deux observables par réplique, en
//! sommes brutes F64 à arbre fixe : S_tot (non signée) et S_feuille ; S_reste = S_tot − S_feuille.
//!
//! Il RÉUTILISE les noyaux de `su2_geo.cu` (voisins, agrafe, surrelaxation) sans les modifier ; `su2_feuille.cu` ajoute un bain
//! thermique identique à β > 0 (et de Haar à β = 0) et le noyau des deux sommes. Aucun type existant n'est modifié.
//!
//! Conventions : s = ((x0·ℓ1 + x1)·ℓ2 + x2)·ℓ3 + x3, lien li = μ·V + s, quaternion (x, y, z, w) ; une réplique = une chaîne du lot
//! (β, graine, départ propres) ; une graine identique pour deux lots (poids +1 et −1) = mode APPARIÉ (mêmes compteurs Philox).

use super::torsion::Torsion;
use super::ParamChaine;
use crate::commun::{BackendKind, Depart, Erreur, Precision, Resultat};
use crate::geom::{Coloriage, GeoA, GeoSpec, Tables};
use crate::gpu::{
    assembler, Appareil, Arg, Fonction, Lancement, Module, OptionsCompil, SourceAssemblee, Tampon,
};
use crate::noyaux::{Fichier, SU2_COMMUN, SU2_REF64};

pub static SU2_FEUILLE32: Fichier = Fichier {
    nom: "su2_feuille32.cu",
    contenu: include_str!("../../noyaux/su2_feuille32.cu"),
};

/// Noyau de l'estimateur à lien intégré (option `--lien-integre` : module à part, absent des modules sans l'option).
pub static SU2_LIEN: Fichier = Fichier {
    nom: "su2_lien.cu",
    contenu: include_str!("../../noyaux/su2_lien.cu"),
};

pub static SU2_FEUILLE: Fichier = Fichier {
    nom: "su2_feuille.cu",
    contenu: include_str!("../../noyaux/su2_feuille.cu"),
};

const BS: u32 = 256;
/// Plafond de chaînes par lot (blockIdx.y < 65536).
pub const CHAINES_MAX: usize = 65535;

/// Source du module de la campagne.
pub fn source_boite(backend: BackendKind) -> SourceAssemblee {
    source_boite_mutant(backend, 0)
}

/// Source avec un mutant de noyau (1 à 5 : `KVU_GEO_MUTANT` de `su2_geo.cu`), CONTRÔLES NÉGATIFS seulement, jamais en production.
pub fn source_boite_mutant(backend: BackendKind, mutant: u8) -> SourceAssemblee {
    let valeur = mutant.to_string();
    let defines = [("KVU_LISTE_CLASSES", "1"), ("KVU_GEO_MUTANT", valeur.as_str())];
    let defines = if mutant > 0 { &defines[..] } else { &defines[..1] };
    assembler(
        backend,
        "su2_boite_une",
        &[
            &SU2_COMMUN,
            &SU2_REF64,
            &super::geo_gpu::SU2_GEO,
            &SU2_FEUILLE,
        ],
        defines,
    )
}

/// Source du chemin rapide F32 (noyaux float, mesures en double).
pub fn source_boite32(backend: BackendKind) -> SourceAssemblee {
    assembler(
        backend,
        "su2_boite_une32",
        &[
            &SU2_COMMUN,
            &crate::noyaux::SU2_F32,
            &SU2_REF64,
            &super::geo_gpu::SU2_GEO,
            &SU2_FEUILLE,
            &SU2_FEUILLE32,
        ],
        &[("KVU_LISTE_CLASSES", "1")],
    )
}

/// Source avec le noyau à lien intégré (mêmes fichiers que `source_boite` / `source_boite32`, plus `su2_lien.cu`).
pub fn source_boite_li(backend: BackendKind, f32: bool) -> SourceAssemblee {
    if f32 {
        assembler(
            backend,
            "su2_boite_une32_li",
            &[&SU2_COMMUN, &crate::noyaux::SU2_F32, &SU2_REF64, &super::geo_gpu::SU2_GEO, &SU2_FEUILLE, &SU2_FEUILLE32, &SU2_LIEN],
            &[("KVU_LISTE_CLASSES", "1"), ("KVU_LI_F32", "1")],
        )
    } else {
        assembler(
            backend,
            "su2_boite_une_li",
            &[&SU2_COMMUN, &SU2_REF64, &super::geo_gpu::SU2_GEO, &SU2_FEUILLE, &SU2_LIEN],
            &[("KVU_LISTE_CLASSES", "1")],
        )
    }
}

/// Source des noyaux à COUTURE décalée : mêmes fichiers, `KVU_COUTURE` défini (voisins du tour de l'axe recollé translatés du décalage).
/// Les sources SANS couture (`source_boite_mutant`, `source_boite32`) restent exactement ce qu'elles étaient.
pub fn source_boite_couture(backend: BackendKind, f32: bool) -> SourceAssemblee {
    let defines = [("KVU_LISTE_CLASSES", "1"), ("KVU_COUTURE", "1")];
    if f32 {
        assembler(
            backend,
            "su2_boite_une32_couture",
            &[&SU2_COMMUN, &crate::noyaux::SU2_F32, &SU2_REF64, &super::geo_gpu::SU2_GEO, &SU2_FEUILLE, &SU2_FEUILLE32],
            &defines,
        )
    } else {
        assembler(
            backend,
            "su2_boite_une_couture",
            &[&SU2_COMMUN, &SU2_REF64, &super::geo_gpu::SU2_GEO, &SU2_FEUILLE],
            &defines,
        )
    }
}

/// Paramètres d'un lot.
#[derive(Clone, Debug)]
pub struct ParamBoite {
    pub cotes: [usize; 4],
    pub torsion: Torsion,
    /// Poids de la feuille (−1 : torsion ; +1 : témoin ; tout réel admis).
    pub poids: f64,
    pub nor: usize,
    pub chaines: Vec<ParamChaine>,
    /// F64 (référence) ou F32 (rapide : noyaux float, mesures et sommes en double).
    pub precision: Precision,
    /// Autorise les côtés IMPAIRS (≥ 3) : coloriage à trois classes (cahier O4 b). Faux : côtés pairs seulement (message historique).
    /// Les côtés pairs ne sont pas affectés (damier à deux classes, mêmes octets de sortie).
    pub impairs: bool,
    /// Réseau anisotrope (A4) : `Some((ξ₀, direction du temps))`. Les agrafes des plaquettes temporelles pèsent ξ₀ et celles des
    /// plaquettes spatiales 1/ξ₀ (action S = (β/ξ₀) Σ_spatiales + β ξ₀ Σ_temporelles). `None` (défaut) : comportement historique,
    /// AUCUN coefficient de plus (même octets). `Some((1.0, _))` est traité comme `None`.
    pub aniso: Option<(f64, usize)>,
    /// Couture décalée (tore oblique) : `Some((axe, décalage))`, voir `GeoA::avec_couture`. `None` (défaut) : réseau ordinaire, source de noyaux
    /// et tampon `geo` INCHANGÉS (mêmes octets). Refusé avec torsion, anisotropie, impairs non demandés.
    pub couture: Option<(usize, [i64; 4])>,
}

impl ParamBoite {
    pub fn valider(&self) -> Resultat<()> {
        if self.cotes.iter().any(|&c| c < 2 || c % 2 != 0) {
            if !self.impairs {
                return Err(Erreur::Config(format!(
                    "côtés {:?} : le chemin GPU de la campagne exige des côtés pairs ≥ 2 (côtés impairs : option --impairs, coloriage à trois classes)",
                    self.cotes
                )));
            }
            if self.cotes.iter().any(|&c| c < 3 && c != 2) {
                return Err(Erreur::Config(format!("côtés {:?} : côtés ≥ 2 requis", self.cotes)));
            }
        }
        let v: usize = self.cotes.iter().product();
        if 4 * v >= (1usize << 31) {
            return Err(Erreur::Config(format!(
                "volume {v} trop grand (4V < 2³¹ requis : compteur Philox sur 32 bits)"
            )));
        }
        if self.chaines.is_empty() || self.chaines.len() > CHAINES_MAX {
            return Err(Erreur::Config(format!(
                "{} chaînes : 1 à {CHAINES_MAX} attendues",
                self.chaines.len()
            )));
        }
        if let Some((xi0, d)) = self.aniso {
            if !(xi0.is_finite() && xi0 > 0.0) || d > 3 {
                return Err(Erreur::Config("--xi0 fini > 0 et direction du temps 0..3 requis".into()));
            }
        }
        if !self.poids.is_finite() {
            return Err(Erreur::Config("poids de feuille non fini".into()));
        }
        if self
            .chaines
            .iter()
            .any(|c| !c.beta.is_finite() || c.beta < 0.0)
        {
            return Err(Erreur::Config("β fini et ≥ 0 requis".into()));
        }
        Ok(())
    }
}

/// Coefficients c(p, s) et masque de feuille construits des Tables (SANS GPU) : coef = z avec −1 remplacé par `poids`.
/// Chaînes par lot de lancement. `KVU_LOT_CHAINES` : 0 = toutes (comportement historique) ; sinon valeur imposée. Sans variable :
/// le plus grand nombre de chaînes dont les champs tiennent dans `KVU_LOT_L2_PCT` % (défaut 70) du cache L2 du périphérique
/// (CUDA) ; HIP ou L2 illisible : toutes (inchangé). Mesuré : RTX 5060 Ti (L2 32 Mo) et RTX 5090 (L2 96 Mo), optimum voisin de 1 à 0,7 × L2.
fn lot_par_defaut(nch: usize, octets_par_chaine: usize, l2: Option<u64>) -> usize {
    if let Some(l) = std::env::var("KVU_LOT_CHAINES").ok().and_then(|t| t.parse::<usize>().ok()) {
        return if l == 0 { nch } else { l.min(nch).max(1) };
    }
    let pct = std::env::var("KVU_LOT_L2_PCT").ok().and_then(|t| t.parse::<u64>().ok()).unwrap_or(70).clamp(1, 400);
    match l2 {
        Some(l2) if octets_par_chaine > 0 => ((l2 * pct / 100) as usize / octets_par_chaine).clamp(1, nch),
        _ => nch,
    }
}

/// Tampon geo de `UneBoite` : les 9 entrées de `tableau_geo`, puis geo[9] = 0 (damier de parité) ou le nombre de classes, les
/// décalages des classes (4 entrées), et la liste des sites triés par classe (côtés impairs). Voir `su2_geo.cu`.
pub fn tableau_geo_classes(t: &Tables) -> Vec<i64> {
    let mut g: Vec<i64> = super::geo_gpu::tableau_geo(t).to_vec();
    if t.coloriage == Coloriage::Parite {
        g.extend_from_slice(&[0, 0, 0, 0, 0]);
        return g;
    }
    assert!(t.classes.len() <= 3);
    g.push(t.classes.len() as i64);
    let mut off = 0i64;
    for k in 0..4 {
        g.push(off);
        if k < t.classes.len() {
            off += t.classes[k].len() as i64;
        }
    }
    for c in &t.classes {
        g.extend(c.iter().map(|&s| s as i64));
    }
    g
}

/// Poids de plan (ordre `PLANS` = 01, 02, 03, 12, 13, 23) du réseau anisotrope : ξ₀ si le plan contient la direction du temps,
/// 1/ξ₀ sinon (le β global reste dehors, comme dans `coef`).
pub fn poids_plans_aniso(xi0: f64, temps: usize) -> [f64; 6] {
    let mut w = [1.0 / xi0; 6];
    for (i, &(mu, nu)) in crate::geom::PLANS.iter().enumerate() {
        if mu == temps || nu == temps {
            w[i] = xi0;
        }
    }
    w
}

/// Coloriage à 3 classes d'un tore oblique par DSATUR avec retour arrière (budget borné) ; `None` si introuvable.
/// Les classes sont rendues triées (sites croissants), comme `GeoA::couleurs`.
fn coloriage_trois(g: &GeoA) -> Option<Vec<Vec<usize>>> {
    let v = g.v;
    let mut vois: Vec<Vec<usize>> = vec![Vec::new(); v];
    for s in 0..v {
        for mu in 0..4 {
            let t = g.decaler(s, mu, 1); // voisin haut (suit la couture) ; le bas est son inverse
            vois[s].push(t);
            vois[t].push(s);
        }
    }
    for n in vois.iter_mut() {
        n.sort_unstable();
        n.dedup();
    }
    if vois.iter().enumerate().any(|(s, n)| n.contains(&s)) {
        return None;
    }
    let mut col = vec![u8::MAX; v];
    let mut budget = 2_000_000u64;
    fn rec(vois: &[Vec<usize>], col: &mut [u8], restant: usize, budget: &mut u64) -> bool {
        if restant == 0 {
            return true;
        }
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        // site non colorié de plus grande saturation (puis de plus grand degré)
        let mut best = (usize::MAX, 0usize, 0usize);
        for s in 0..col.len() {
            if col[s] != u8::MAX {
                continue;
            }
            let mut m = 0u8;
            for &n in &vois[s] {
                if col[n] != u8::MAX {
                    m |= 1 << col[n];
                }
            }
            let sat = m.count_ones() as usize;
            if best.0 == usize::MAX || sat > best.1 || (sat == best.1 && vois[s].len() > best.2) {
                best = (s, sat, vois[s].len());
            }
        }
        let s = best.0;
        let mut m = 0u8;
        for &n in &vois[s] {
            if col[n] != u8::MAX {
                m |= 1 << col[n];
            }
        }
        for k in 0..3u8 {
            if m & (1 << k) == 0 {
                col[s] = k;
                if rec(vois, col, restant - 1, budget) {
                    return true;
                }
                col[s] = u8::MAX;
            }
        }
        false
    }
    if !rec(&vois, &mut col, v, &mut budget) {
        return None;
    }
    let mut cl = vec![Vec::new(); 3];
    for (s, &c) in col.iter().enumerate() {
        cl[c as usize].push(s);
    }
    cl.retain(|c| !c.is_empty());
    Some(cl)
}

/// Réseau à couture décalée de `p` (`None` si `p.couture` est absente ou si le décalage est nul modulo les côtés : réseau ordinaire).
pub fn geo_couture(p: &ParamBoite) -> Resultat<Option<GeoA>> {
    match p.couture {
        None => Ok(None),
        Some((axe, vec)) => {
            let g = GeoA::avec_couture(p.cotes, axe, vec)?;
            Ok(g.couture.is_some().then_some(g))
        }
    }
}

pub fn tables_boite(p: &ParamBoite) -> Resultat<(Tables, Option<Vec<f64>>, Vec<u8>)> {
    tables_boite_fixe(p, None)
}

/// Union de deux torsions (plans tordus de l'une OU de l'autre), sans toucher à `geom`.
fn union_torsions(a: &Torsion, b: &Torsion) -> Resultat<Torsion> {
    let l: Vec<String> = Torsion::PLANS
        .iter()
        .filter(|&&(m, n)| a.tordu(m, n) || b.tordu(m, n))
        .map(|&(m, n)| format!("{m}{n}"))
        .collect();
    if l.is_empty() {
        Ok(Torsion::triviale())
    } else {
        Torsion::analyser(&l.join(","))
    }
}

/// Comme `tables_boite`, avec des plans de torsion FIXES (`fixe`) : leur pile garde le coefficient −1 quel que soit `p.poids` et
/// n'entre pas dans le masque de feuille ; `p.torsion` garde les plans à poids variable. Un plan dans les deux listes est refusé.
/// `fixe = None` ou vide : exactement `tables_boite` historique (mêmes octets).
pub fn tables_boite_fixe(p: &ParamBoite, fixe: Option<&Torsion>) -> Resultat<(Tables, Option<Vec<f64>>, Vec<u8>)> {
    tables_boite_gen(p, fixe, None)
}

/// Poids des six plans (ordre `PLANS`) pour des couplages d'espace et de temps SÉPARÉS : `(ws, wt)` ; le temps est la direction 0
/// (plans 01, 02, 03). Le β de chaîne multiplie ces poids : on prend β de chaîne = β_s, ws = 1, wt = β_t/β_s (wt = 0 admis : β_t = 0).
pub fn poids_plans_st(ws: f64, wt: f64) -> [f64; 6] {
    let mut w = [ws; 6];
    for (i, &(mu, nu)) in crate::geom::PLANS.iter().enumerate() {
        if mu == 0 || nu == 0 {
            w[i] = wt;
        }
    }
    w
}

/// `tables_boite_fixe` + option `st = Some((ws, wt))` : couplages d'espace et de temps séparés (sans torsion). Le coefficient de
/// chaque plan est ws (plans spatiaux) ou wt (plans 0i) ; le masque de feuille marque les plaquettes TEMPORELLES, de sorte que
/// `S_feuille` = S_t (le noyau de mesure est inchangé). `st = None` : exactement l'historique (mêmes octets).
pub fn tables_boite_gen(
    p: &ParamBoite,
    fixe: Option<&Torsion>,
    st: Option<(f64, f64)>,
) -> Resultat<(Tables, Option<Vec<f64>>, Vec<u8>)> {
    p.valider()?;
    if let Some((ws, wt)) = st {
        if !p.torsion.est_triviale() || fixe.is_some_and(|f| !f.est_triviale()) || p.aniso.is_some() || p.couture.is_some() {
            return Err(Erreur::Config(
                "couplages d'espace et de temps séparés : refusés avec une torsion, --xi0 ou une couture décalée (la feuille sert aux plaquettes temporelles)".into(),
            ));
        }
        if !(ws.is_finite() && ws >= 0.0 && wt.is_finite() && wt >= 0.0) {
            return Err(Erreur::Config("poids d'espace/temps finis et ≥ 0 requis".into()));
        }
        let t = Tables::compiler(&GeoSpec::anisotrope(p.cotes, 3))?;
        if t.coloriage != Coloriage::Parite && !p.impairs {
            return Err(Erreur::Config("coloriage non pair : refusé (option impairs requise)".into()));
        }
        let v = t.v;
        let w = poids_plans_st(ws, wt);
        let mut coef = vec![0.0f64; 6 * v];
        let mut feuille = vec![0u8; 6 * v];
        for (i, &(mu, nu)) in crate::geom::PLANS.iter().enumerate() {
            coef[i * v..(i + 1) * v].fill(w[i]);
            if mu == 0 || nu == 0 {
                feuille[i * v..(i + 1) * v].fill(1);
            }
        }
        return Ok((t, Some(coef), feuille));
    }
    let fixe = fixe.filter(|f| !f.est_triviale());
    let mut torsion = p.torsion;
    if let Some(f) = fixe {
        if let Some(&(m, n)) = Torsion::PLANS.iter().find(|&&(m, n)| f.tordu(m, n) && p.torsion.tordu(m, n)) {
            return Err(Erreur::Config(format!(
                "plan {m}{n} à la fois dans --torsion-fixe et --torsion : refusé"
            )));
        }
        torsion = union_torsions(&p.torsion, f)?;
    }
    let mut spec = GeoSpec::anisotrope(p.cotes, 3).avec_torsion(torsion);
    if let Some((xi0, d)) = p.aniso.filter(|&(x, _)| x != 1.0) {
        spec = spec.avec_beta_plans(poids_plans_aniso(xi0, d));
    }
    let mut t = Tables::compiler(&spec)?;
    if let Some(g) = geo_couture(p)? {
        // Couture décalée : classes du tore oblique (vérifiées, au plus 3 : limite du tampon `geo`), liste explicite de sites.
        if !torsion.est_triviale() || p.aniso.is_some() || t.coef.is_some() {
            return Err(Erreur::Config("couture décalée : sans torsion ni anisotropie (non prises en charge)".into()));
        }
        let mut cl = g.couleurs();
        if cl.len() > 3 {
            cl = coloriage_trois(&g).ok_or_else(|| Erreur::Config(format!("couture : pas de coloriage à 3 classes trouvé ({} par le glouton ; 3 au plus sur carte)", cl.len())))?;
        }
        g.verifier_classes(&cl).map_err(|e| Erreur::Config(format!("couture : coloriage invalide : {e}")))?;
        t.classes = cl;
        t.coloriage = Coloriage::Glouton;
    }
    if t.coloriage != Coloriage::Parite && !p.impairs {
        return Err(Erreur::Config("coloriage non pair : refusé (option impairs requise)".into()));
    }
    let v = t.v;
    let mut feuille = vec![0u8; 6 * v];
    let coef = t.coef.as_ref().map(|c| {
        c.iter()
            .enumerate()
            .map(|(i, &z)| {
                if z == -1.0 {
                    let (m, n) = Torsion::PLANS[i / v];
                    if fixe.is_some_and(|f| f.tordu(m, n)) {
                        z
                    } else {
                        feuille[i] = 1;
                        p.poids
                    }
                } else {
                    z
                }
            })
            .collect::<Vec<f64>>()
    });
    Ok((t, coef, feuille))
}

struct LiEtat {
    f: Fonction,
    site: [Tampon<f64>; 3],
    hist: Option<Tampon<f64>>,
}

/// Une boîte, un lot de répliques, sur un GPU (HIP ou CUDA), F64.
pub struct UneBoite {
    app: Appareil,
    p: ParamBoite,
    v: usize,
    nch: usize,
    _module: Module,
    f_init: Fonction,
    f_hb: Fonction,
    f_or: Fonction,
    f_site: Fonction,
    f_reduce: Fonction,
    f_final: Fonction,
    geo: Tampon<i64>,
    /// Nombre de classes de sites du balayage : 2 (damier) ou 3 (côtés impairs).
    ncl: i32,
    /// Plus grande classe (taille de grille des lancements de bain et de surrelaxation).
    max_classe: usize,
    coef: Option<Tampon<f64>>,
    /// F32 : copie float du coefficient (conversion IEEE au plus proche, identique à `(float)coef[i]` sur le GPU).
    coef32: Option<Tampon<f32>>,
    feuille: Tampon<u8>,
    /// Masque de liens gelés (algorithme à deux niveaux) : octet par lien local li = μ·V + s, partagé par toutes les chaînes ; `None` = aucun gel.
    masque: Option<Tampon<u8>>,
    champ: Tampon<f64>,
    champ32: Option<Tampon<f32>>,
    betas: Tampon<f64>,
    k0s: Tampon<u32>,
    k1s: Tampon<u32>,
    hot: Tampon<i32>,
    cap: Tampon<u64>,
    tries: Tampon<u64>,
    site_tot: Tampon<f64>,
    site_fe: Tampon<f64>,
    part: Tampon<f64>,
    hist: Option<(Tampon<f64>, usize)>,
    /// Lien intégré (option) : noyau, trois tableaux par site (tot, t, s), historique [chaîne][idx][3].
    li: Option<LiEtat>,
    sweep_id: u32,
    /// Chaînes par lot de lancement (cache L2) : un lot enchaîne ses 8 lancements de bain puis ceux de surrelaxation avant le
    /// lot suivant. Les chaînes sont indépendantes (clé Philox, β, champ propres) : résultat identique au bit à `lot = nch`.
    lot: usize,
    pub empreinte_source: String,
}

impl UneBoite {
    pub fn nouveau(app: &Appareil, p: &ParamBoite) -> Resultat<UneBoite> {
        Self::nouveau_mutant(app, p, 0)
    }

    /// Contrôle négatif (mutant de noyau 1 à 5) ; `mutant = 0` : production.
    #[doc(hidden)]
    pub fn nouveau_mutant(app: &Appareil, p: &ParamBoite, mutant: u8) -> Resultat<UneBoite> {
        Self::nouveau_gen(app, p, mutant, None, None, None, false)
    }

    /// Boîte avec plans de torsion FIXES (`fixe`, coefficient −1 quel que soit `p.poids`, hors masque de feuille) : voir `tables_boite_fixe`.
    pub fn nouveau_fixe(app: &Appareil, p: &ParamBoite, fixe: &Torsion) -> Resultat<UneBoite> {
        Self::nouveau_gen(app, p, 0, None, Some(*fixe), None, false)
    }

    /// Couplages d'espace et de temps SÉPARÉS (`ws`, `wt` : voir `tables_boite_gen`) ; `S_feuille` des séries = S_t. Sans torsion.
    pub fn nouveau_st(app: &Appareil, p: &ParamBoite, ws: f64, wt: f64) -> Resultat<UneBoite> {
        Self::nouveau_gen(app, p, 0, None, None, Some((ws, wt)), false)
    }

    /// Comme `nouveau` / `nouveau_st` (`st = Some((ws, wt))` ou isotrope) avec, si `li`, l'estimateur à lien intégré (`serie_li`).
    /// Sans torsion, sans couture, sans mutant.
    pub fn nouveau_li(app: &Appareil, p: &ParamBoite, st: Option<(f64, f64)>) -> Resultat<UneBoite> {
        Self::nouveau_gen(app, p, 0, None, None, st, true)
    }

    /// TESTS seulement : impose les classes de sites du balayage (listes triées par classe ; 1 à 3 classes), pour les tests
    /// d'ordre intérieur (classes mélangées : résultat identique au bit) et les mutants de coloriage (I7).
    #[doc(hidden)]
    pub fn nouveau_classes(app: &Appareil, p: &ParamBoite, classes: Vec<Vec<usize>>) -> Resultat<UneBoite> {
        Self::nouveau_gen(app, p, 0, Some(classes), None, None, false)
    }

    fn nouveau_gen(app: &Appareil, p: &ParamBoite, mutant: u8, classes_forcees: Option<Vec<Vec<usize>>>, fixe: Option<Torsion>, st: Option<(f64, f64)>, li: bool) -> Resultat<UneBoite> {
        if mutant > 5 {
            return Err(Erreur::Config(format!("mutant {mutant} : 0 à 5")));
        }
        let (mut t, coef_h, feuille_h) = tables_boite_gen(p, fixe.as_ref(), st)?;
        if let Some(c) = classes_forcees {
            if c.is_empty() || c.len() > 3 {
                return Err(Erreur::Config("classes forcées : 1 à 3".into()));
            }
            t.classes = c;
            t.coloriage = Coloriage::Glouton;
        }
        let (v, nch) = (t.v, p.chaines.len());
        let nb = v.div_ceil(BS as usize);
        let besoin = 8
            * (nch as u64 * (16 * v as u64 + 2 * v as u64 + nb as u64 + 8) + 6 * v as u64)
            + 6 * v as u64;
        if let Ok(libre) = app.memoire_libre_octets() {
            if besoin > libre / 10 * 9 {
                return Err(Erreur::Ressources(format!(
                    "{nch} répliques de {:?} : {:.2} Go nécessaires, {:.2} Go libres sur la carte",
                    p.cotes,
                    besoin as f64 / 1e9,
                    libre as f64 / 1e9
                )));
            }
        }
        let f32 = p.precision == Precision::F32;
        if f32 && mutant != 0 {
            return Err(Erreur::Config("mutants de noyau : F64 seulement".into()));
        }
        let couture = geo_couture(p)?;
        if couture.is_some() && mutant != 0 {
            return Err(Erreur::Config("couture : sans mutant de noyau".into()));
        }
        if li && (couture.is_some() || mutant != 0 || !p.torsion.est_triviale() || fixe.is_some()) {
            return Err(Erreur::Config(
                "lien intégré : refusé avec une torsion, --torsion-fixe, une couture décalée (S_feuille à lien intégré : non fait)".into(),
            ));
        }
        let src = match (couture.is_some(), f32) {
            _ if li => source_boite_li(app.backend(), f32),
            (true, true) => source_boite_couture(app.backend(), true),
            (true, false) => source_boite_couture(app.backend(), false),
            (false, true) => source_boite32(app.backend()),
            (false, false) => source_boite_mutant(app.backend(), mutant),
        };
        let module = app.compiler(
            &src,
            &if f32 {
                OptionsCompil::rapide()
            } else {
                OptionsCompil::reference_k1()
            },
        )?;
        let f = |n: &str| app.fonction(&module, n);
        let mut geo_h = tableau_geo_classes(&t);
        if let Some(g) = &couture {
            // Couture : [axe, décalage ×4] derrière la liste des classes (offset 14 + geo[13], voir su2_geo.cu).
            let c = g.couture.as_ref().unwrap();
            geo_h.push(c.axe as i64);
            geo_h.extend(c.vec.iter().map(|&x| x as i64));
        }
        let mut geo = app.alloc::<i64>(geo_h.len())?;
        app.ecrire(&mut geo, 0, &geo_h)?;
        let ncl: i32 = if t.coloriage == Coloriage::Parite { 2 } else { t.classes.len() as i32 };
        let taille_max_classe = if t.coloriage == Coloriage::Parite { v / 2 } else { t.classes.iter().map(Vec::len).max().unwrap_or(0) };
        let coef32 = match (&coef_h, f32) {
            (Some(c), true) => {
                let mut b = app.alloc::<f32>(c.len())?;
                app.ecrire(&mut b, 0, &c.iter().map(|&z| z as f32).collect::<Vec<f32>>())?;
                Some(b)
            }
            _ => None,
        };
        let coef = match &coef_h {
            Some(_) if f32 => None,
            Some(c) => {
                let mut b = app.alloc::<f64>(c.len())?;
                app.ecrire(&mut b, 0, c)?;
                Some(b)
            }
            None => None,
        };
        let mut feuille = app.alloc::<u8>(6 * v)?;
        app.ecrire(&mut feuille, 0, &feuille_h)?;
        let mut betas = app.alloc::<f64>(nch)?;
        let mut k0s = app.alloc::<u32>(nch)?;
        let mut k1s = app.alloc::<u32>(nch)?;
        let mut hot = app.alloc::<i32>(nch)?;
        let cles: Vec<(u32, u32)> = p.chaines.iter().map(|c| c.graine.cle()).collect();
        app.ecrire(
            &mut betas,
            0,
            &p.chaines.iter().map(|c| c.beta).collect::<Vec<_>>(),
        )?;
        app.ecrire(&mut k0s, 0, &cles.iter().map(|k| k.0).collect::<Vec<_>>())?;
        app.ecrire(&mut k1s, 0, &cles.iter().map(|k| k.1).collect::<Vec<_>>())?;
        app.ecrire(
            &mut hot,
            0,
            &p.chaines
                .iter()
                .map(|c| i32::from(c.depart == Depart::Chaud))
                .collect::<Vec<_>>(),
        )?;
        Ok(UneBoite {
            f_init: f(if f32 { "k_init_f32g" } else { "k_init_geo" })?,
            f_hb: f(if f32 {
                "k_heatbath_f32g"
            } else {
                "k_heatbath_boite"
            })?,
            f_or: f(if f32 {
                "k_overrelax_f32g"
            } else {
                "k_overrelax_geo"
            })?,
            f_site: f(if f32 {
                if std::env::var("KVU_MESURE_F32").is_ok_and(|v| v == "1") {
                    "k_plaq_site_feuille_f32gs" // option non identique au bit (désactivée par défaut)
                } else {
                    "k_plaq_site_feuille_f32g"
                }
            } else {
                "k_plaq_site_feuille"
            })?,
            f_reduce: f("k_reduce")?,
            f_final: f("k_final")?,
            champ: app.alloc::<f64>(if f32 { 1 } else { nch * 16 * v })?,
            champ32: if f32 {
                Some(app.alloc::<f32>(nch * 16 * v)?)
            } else {
                None
            },
            cap: app.alloc::<u64>(nch)?,
            tries: app.alloc::<u64>(nch)?,
            site_tot: app.alloc::<f64>(nch * v)?,
            site_fe: app.alloc::<f64>(nch * v)?,
            part: app.alloc::<f64>(nch * nb)?,
            geo,
            ncl,
            max_classe: taille_max_classe,
            coef,
            coef32,
            feuille,
            masque: None,
            betas,
            k0s,
            k1s,
            hot,
            hist: None,
            li: if li {
                let nv = nch * v;
                Some(LiEtat {
                    f: f("k_plaq_site_li")?,
                    site: [app.alloc::<f64>(nv)?, app.alloc::<f64>(nv)?, app.alloc::<f64>(nv)?],
                    hist: None,
                })
            } else {
                None
            },
            sweep_id: 0,
            lot: lot_par_defaut(nch, 16 * v * 4, app.cache_l2_octets()),
            empreinte_source: src.sha256.clone(),
            app: app.clone(),
            p: p.clone(),
            v,
            nch,
            _module: module,
        })
    }

    pub fn volume(&self) -> usize {
        self.v
    }
    /// Accès pour la spectroscopie sur carte (spectro/gpu-1120) : champ de TOUTES les chaînes (4V quaternions chacune)
    /// et tableau de géométrie.
    pub(crate) fn arg_champ_complet(&self) -> Arg {
        self.champ_arg()
    }
    pub(crate) fn arg_geo(&self) -> Arg {
        self.geo.arg()
    }
    pub fn param(&self) -> &ParamBoite {
        &self.p
    }
    pub fn nchaines(&self) -> usize {
        self.nch
    }
    pub fn sweep_id(&self) -> u32 {
        self.sweep_id
    }

    /// Départ froid ou chaud (par chaîne), compteur de balayages remis à zéro.
    pub fn init(&mut self) -> Resultat<()> {
        self.app.lancer(
            &self.f_init,
            &Lancement::lineaire_lot(4 * self.v, self.nch, BS),
            &[
                self.champ_arg(),
                self.geo.arg(),
                self.hot.arg(),
                self.k0s.arg(),
                self.k1s.arg(),
            ],
        )?;
        self.app.synchroniser()?;
        let z = vec![0u64; self.nch];
        self.app.ecrire(&mut self.cap, 0, &z)?;
        self.app.ecrire(&mut self.tries, 0, &z)?;
        self.sweep_id = 0;
        Ok(())
    }

    fn champ_arg(&self) -> Arg {
        self.champ32
            .as_ref()
            .map_or_else(|| self.champ.arg(), |c| c.arg())
    }

    fn coef_arg(&self) -> Arg {
        if self.champ32.is_some() {
            return self.coef32.as_ref().map_or(Arg::Nul, |c| c.arg());
        }
        self.coef.as_ref().map_or(Arg::Nul, |c| c.arg())
    }

    /// Argument du champ décalé de `c0` chaînes (F32 : 16 flottants par site et par chaîne ; F64 : idem en doubles).
    fn champ_arg_dec(&self, c0: usize) -> Arg {
        match &self.champ32 {
            Some(c) => c.arg_decale(c0 * 16 * self.v),
            None => self.champ.arg_decale(c0 * 16 * self.v),
        }
    }

    /// Plages (première chaîne, nombre) des lots de lancement.
    fn lots(&self) -> Vec<(usize, usize)> {
        let l = self.lot.clamp(1, self.nch.max(1));
        (0..self.nch).step_by(l).map(|c0| (c0, l.min(self.nch - c0))).collect()
    }

    /// Règle le nombre de chaînes par lot de lancement (0 ou ≥ nch : un seul lot, comportement historique).
    pub fn regler_lot(&mut self, lot: usize) {
        self.lot = if lot == 0 { self.nch } else { lot.min(self.nch) };
    }

    pub fn lot(&self) -> usize {
        self.lot
    }

    fn args_hb(&self, mu: i32, par: i32, c0: usize) -> Vec<Arg> {
        let mut a = vec![
            self.champ_arg_dec(c0),
            self.geo.arg(),
            Arg::I32(mu),
            Arg::I32(par),
            self.betas.arg_decale(c0),
            self.k0s.arg_decale(c0),
            self.k1s.arg_decale(c0),
            Arg::U32(self.sweep_id),
            self.cap.arg_decale(c0),
            self.tries.arg_decale(c0),
        ];
        if self.champ32.is_none() {
            a.push(Arg::Nul); // diagnostic des essais
            a.push(self.coef_arg());
            a.push(self.masque_arg()); // masque
        } else {
            a.push(self.coef_arg());
            a.push(self.masque_arg());
        }
        a
    }

    fn args_or(&self, mu: i32, par: i32, c0: usize) -> Vec<Arg> {
        // les deux branches (champ F32 ou non) donnaient la même liste ; fusionnées (style seulement)
        vec![
            self.champ_arg_dec(c0),
            self.geo.arg(),
            Arg::I32(mu),
            Arg::I32(par),
            self.coef_arg(),
            self.masque_arg(),
        ]
    }

    fn masque_arg(&self) -> Arg {
        match &self.masque {
            Some(t) => t.arg(),
            None => Arg::Nul,
        }
    }

    /// Pose (ou retire, `None`) le masque de liens gelés : 4·V octets (li = μ·V + s), un octet non nul gèle le lien (ni bain
    /// thermique ni surrelaxation). Sans masque, sorties identiques au bit.
    pub fn fixer_masque(&mut self, m: Option<&[u8]>) -> Resultat<()> {
        match m {
            None => self.masque = None,
            Some(m) => {
                if m.len() != 4 * self.v {
                    return Err(Erreur::Config(format!("masque de {} octets : {} attendus (4·V)", m.len(), 4 * self.v)));
                }
                let mut t = self.app.alloc::<u8>(m.len())?;
                self.app.ecrire(&mut t, 0, m)?;
                self.app.synchroniser()?;
                self.masque = Some(t);
            }
        }
        Ok(())
    }

    fn bain_lot(&self, c0: usize, n: usize) -> Resultat<()> {
        let cfg = Lancement::lineaire_lot(self.max_classe, n, BS);
        for mu in 0..4i32 {
            for par in 0..self.ncl {
                self.app.lancer(&self.f_hb, &cfg, &self.args_hb(mu, par, c0))?;
            }
        }
        Ok(())
    }

    fn surrelax_lot(&self, c0: usize, n: usize) -> Resultat<()> {
        let cfg = Lancement::lineaire_lot(self.max_classe, n, BS);
        for mu in 0..4i32 {
            for par in 0..self.ncl {
                self.app.lancer(&self.f_or, &cfg, &self.args_or(mu, par, c0))?;
            }
        }
        Ok(())
    }

    /// Un bain thermique (4 μ × 2 parités).
    pub fn sweep_hb(&mut self) -> Resultat<()> {
        for (c0, n) in self.lots() {
            self.bain_lot(c0, n)?;
        }
        self.sweep_id = self.sweep_id.wrapping_add(1);
        Ok(())
    }

    /// Un balayage de surrelaxation.
    pub fn sweep_or(&mut self) -> Resultat<()> {
        for (c0, n) in self.lots() {
            self.surrelax_lot(c0, n)?;
        }
        Ok(())
    }

    /// Un pas = 1 bain thermique + `nor` surrelaxations. Chaque lot de chaînes enchaîne bain puis surrelaxations
    /// (le champ du lot reste dans le cache L2) ; les chaînes étant indépendantes, la sortie est celle du parcours par balayage.
    pub fn pas(&mut self) -> Resultat<()> {
        for (c0, n) in self.lots() {
            self.pas_lot(c0, n)?;
        }
        self.sweep_id = self.sweep_id.wrapping_add(1);
        Ok(())
    }

    /// Un pas pour les chaînes `c0..c0+n` seulement, avec le compteur de balayages courant (non incrémenté).
    fn pas_lot(&self, c0: usize, n: usize) -> Resultat<()> {
        self.bain_lot(c0, n)?;
        for _ in 0..self.p.nor {
            self.surrelax_lot(c0, n)?;
        }
        Ok(())
    }

    /// Alloue l'historique : `capacite` mesures × 2 sommes (S_tot, S_feuille) par chaîne.
    pub fn allouer_historique(&mut self, capacite: usize) -> Resultat<()> {
        let t = self.app.alloc::<f64>(self.nch * capacite * 2)?;
        self.hist = Some((t, capacite));
        if self.li.is_some() {
            let h = self.app.alloc::<f64>(self.nch * capacite * 3)?;
            if let Some(l) = self.li.as_mut() {
                l.hist = Some(h);
            }
        }
        self.app.synchroniser()
    }

    /// Range (S_tot, S_feuille) de chaque chaîne à la case `idx` de l'historique, sans synchronisation.
    pub fn mesurer(&mut self, idx: usize) -> Resultat<()> {
        for (c0, n) in self.lots() {
            self.mesurer_lot(idx, c0, n)?;
        }
        Ok(())
    }

    /// Mesure des chaînes `c0..c0+n` (mêmes noyaux, mêmes arbres de réduction : chaque chaîne est réduite indépendamment).
    fn mesurer_lot(&self, idx: usize, c0: usize, n: usize) -> Resultat<()> {
        let (hist, cap) = self
            .hist
            .as_ref()
            .ok_or_else(|| Erreur::Config("historique non alloué".into()))?;
        if idx >= *cap {
            return Err(Erreur::Config(format!("historique plein : {idx} ≥ {cap}")));
        }
        let v = self.v;
        let nb = v.div_ceil(BS as usize);
        let un_bloc = Lancement {
            grille: [1, n as u32, 1],
            bloc: [BS, 1, 1],
            shmem: 0,
        };
        let pas_h = Arg::I64((2 * *cap) as i64);
        self.app.lancer(
            &self.f_site,
            &Lancement::lineaire_lot(v, n, BS),
            &[
                self.champ_arg_dec(c0),
                self.geo.arg(),
                self.feuille.arg(),
                self.site_tot.arg_decale(c0 * v),
                self.site_fe.arg_decale(c0 * v),
            ],
        )?;
        for (k, site) in [&self.site_tot, &self.site_fe].into_iter().enumerate() {
            self.app.lancer(
                &self.f_reduce,
                &Lancement::lineaire_lot(v, n, BS),
                &[site.arg_decale(c0 * v), Arg::I64(v as i64), self.part.arg_decale(c0 * nb)],
            )?;
            self.app.lancer(
                &self.f_final,
                &un_bloc,
                &[
                    self.part.arg_decale(c0 * nb),
                    Arg::I64(nb as i64),
                    Arg::I64(nb as i64),
                    Arg::F64(1.0),
                    hist.arg_decale(c0 * 2 * *cap + 2 * idx + k),
                    pas_h,
                ],
            )?;
        }
        if let Some(li) = &self.li {
            let hl = li.hist.as_ref().ok_or_else(|| Erreur::Config("historique du lien intégré non alloué".into()))?;
            self.app.lancer(
                &li.f,
                &Lancement::lineaire_lot(v, n, BS),
                &[
                    self.champ_arg_dec(c0),
                    self.geo.arg(),
                    self.coef_arg(),
                    self.betas.arg_decale(c0),
                    li.site[0].arg_decale(c0 * v),
                    li.site[1].arg_decale(c0 * v),
                    li.site[2].arg_decale(c0 * v),
                ],
            )?;
            for (k, site) in li.site.iter().enumerate() {
                self.app.lancer(
                    &self.f_reduce,
                    &Lancement::lineaire_lot(v, n, BS),
                    &[site.arg_decale(c0 * v), Arg::I64(v as i64), self.part.arg_decale(c0 * nb)],
                )?;
                self.app.lancer(
                    &self.f_final,
                    &un_bloc,
                    &[
                        self.part.arg_decale(c0 * nb),
                        Arg::I64(nb as i64),
                        Arg::I64(nb as i64),
                        Arg::F64(1.0),
                        hl.arg_decale(c0 * 3 * *cap + 3 * idx + k),
                        Arg::I64((3 * *cap) as i64),
                    ],
                )?;
            }
        }
        Ok(())
    }

    /// Historique [chaîne][idx][2] (synchronise).
    pub fn historique(&self) -> Resultat<Vec<f64>> {
        let (hist, cap) = self
            .hist
            .as_ref()
            .ok_or_else(|| Erreur::Config("historique non alloué".into()))?;
        self.app.synchroniser()?;
        let mut h = vec![0f64; self.nch * cap * 2];
        self.app.lire(hist, 0, &mut h)?;
        Ok(h)
    }

    /// Historique du lien intégré [chaîne][idx][3] = (S_tot, S_t, S_s) (synchronise).
    pub fn historique_li(&self) -> Resultat<Vec<f64>> {
        let (_, cap) = self.hist.as_ref().ok_or_else(|| Erreur::Config("historique non alloué".into()))?;
        let hl = self
            .li
            .as_ref()
            .and_then(|l| l.hist.as_ref())
            .ok_or_else(|| Erreur::Config("lien intégré non activé".into()))?;
        self.app.synchroniser()?;
        let mut h = vec![0f64; self.nch * cap * 3];
        self.app.lire(hl, 0, &mut h)?;
        Ok(h)
    }

    /// Plafonds atteints par le bain (attendu 0) et essais, par chaîne.
    pub fn compteurs(&self) -> Resultat<Vec<(u64, u64)>> {
        self.app.synchroniser()?;
        let mut c = vec![0u64; self.nch];
        let mut t = vec![0u64; self.nch];
        self.app.lire(&self.cap, 0, &mut c)?;
        self.app.lire(&self.tries, 0, &mut t)?;
        Ok(c.into_iter().zip(t).collect())
    }

    /// Champ brut d'une chaîne (16·V f64, disposition (li·4 + k)).
    pub fn telecharger(&self, chaine: usize) -> Resultat<Vec<f64>> {
        if chaine >= self.nch {
            return Err(Erreur::Config(format!("chaîne {chaine} hors du lot")));
        }
        self.app.synchroniser()?;
        if let Some(c) = &self.champ32 {
            let mut h = vec![0f32; 16 * self.v];
            self.app.lire(c, chaine * 16 * self.v, &mut h)?;
            return Ok(h.into_iter().map(f64::from).collect());
        }
        let mut h = vec![0f64; 16 * self.v];
        self.app.lire(&self.champ, chaine * 16 * self.v, &mut h)?;
        Ok(h)
    }

    /// Pose le champ brut d'une chaîne (tests : mêmes configurations pour les mesures et les balayages).
    pub fn televerser(&mut self, chaine: usize, q: &[f64]) -> Resultat<()> {
        if chaine >= self.nch || q.len() != 16 * self.v {
            return Err(Erreur::Config("champ de taille inattendue".into()));
        }
        if let Some(c) = self.champ32.as_mut() {
            let q32: Vec<f32> = q.iter().map(|&x| x as f32).collect();
            self.app.ecrire(c, chaine * 16 * self.v, &q32)?;
        } else {
            self.app.ecrire(&mut self.champ, chaine * 16 * self.v, q)?;
        }
        self.app.synchroniser()
    }

    /// Série complète d'un run : `therm` pas jetés, puis `nmes` mesures espacées de `skip` pas ; rend, par chaîne,
    /// les `nmes` couples (S_tot, S_feuille). Les mesures sont groupées par tranches pour limiter les synchronisations.
    pub fn serie(
        &mut self,
        therm: usize,
        nmes: usize,
        skip: usize,
    ) -> Resultat<Vec<Vec<[f64; 2]>>> {
        Ok(self.serie_li(therm, nmes, skip)?.0)
    }

    /// Comme `serie`, avec en plus (si la boîte est construite par `nouveau_li`) les séries (S_tot, S_t, S_s) à lien intégré,
    /// mesurées sur LES MÊMES configurations. La première sortie est identique au bit à celle de `serie` sans lien intégré.
    #[allow(clippy::type_complexity)] // style seulement (indices explicites conservés : aucun changement numérique)
    #[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
    pub fn serie_li(
        &mut self,
        therm: usize,
        nmes: usize,
        skip: usize,
    ) -> Resultat<(Vec<Vec<[f64; 2]>>, Option<Vec<Vec<[f64; 3]>>>)> {
        let tranche = nmes.clamp(1, 256);
        let mut sortie: Vec<Vec<[f64; 2]>> = vec![Vec::with_capacity(nmes); self.nch];
        let mut sortie_li: Option<Vec<Vec<[f64; 3]>>> = self.li.as_ref().map(|_| vec![Vec::with_capacity(nmes); self.nch]);
        self.allouer_historique(tranche)?;
        // Chaque lot de chaînes parcourt TOUTE la série (thermalisation, pas, mesures) avant le lot suivant : son champ reste dans le
        // cache L2. Le compteur de balayages repart de la même valeur pour chaque lot ; les chaînes étant indépendantes, la sortie
        // est identique au bit à celle du parcours par balayage sur toutes les chaînes.
        let base = self.sweep_id;
        let mut fin = base;
        for (c0, nl) in self.lots() {
            self.sweep_id = base;
            for _ in 0..therm {
                self.pas_lot(c0, nl)?;
                self.sweep_id = self.sweep_id.wrapping_add(1);
            }
            let mut fait = 0;
            while fait < nmes {
                let n = tranche.min(nmes - fait);
                for i in 0..n {
                    for _ in 0..skip.max(1) {
                        self.pas_lot(c0, nl)?;
                        self.sweep_id = self.sweep_id.wrapping_add(1);
                    }
                    self.mesurer_lot(i, c0, nl)?;
                }
                let h = self.historique()?;
                for c in c0..c0 + nl {
                    for i in 0..n {
                        sortie[c].push([h[(c * tranche + i) * 2], h[(c * tranche + i) * 2 + 1]]);
                    }
                }
                if sortie_li.is_some() {
                    let hl = self.historique_li()?;
                    if let Some(sl) = sortie_li.as_mut() {
                        for c in c0..c0 + nl {
                            for i in 0..n {
                                let j = (c * tranche + i) * 3;
                                sl[c].push([hl[j], hl[j + 1], hl[j + 2]]);
                            }
                        }
                    }
                }
                fait += n;
            }
            fin = self.sweep_id;
        }
        self.sweep_id = fin;
        Ok((sortie, sortie_li))
    }
}
