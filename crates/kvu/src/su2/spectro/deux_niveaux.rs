//! [EXPÉRIMENTAL] Algorithme « à deux niveaux » (Lüscher–Weisz 2001 ; Barca et al., arXiv:2406.12656) pour les corrélateurs
//! spectroscopiques à impulsion nulle.
//!
//! Principe : le temps (axe `temps`, période T, multiple de d) est coupé en tranches épaisses de d mailles. Les liens SPATIAUX
//! des tranches de temps t ≡ 0 (mod d) sont GELÉS ; à chaque configuration externe on fait N₁ mises à jour internes de tout
//! le reste (liens de temps compris), et l'opérateur O_a(t) (lissage APE et lignes de Wilson : liens spatiaux d'une seule
//! tranche) est moyenné sur ces N₁ mises à jour. À frontières fixées, deux tranches épaisses distinctes sont indépendantes :
//! pour (t, t') situés dans deux tranches épaisses différentes, E[O_a(t) O_b(t')] = E_ext[ Ō_a(t) Ō_b(t') ].
//! Un temps gelé (t ≡ 0 mod d) est constant à l'intérieur de la boucle interne : il est compatible avec tout autre temps.
//! Pour deux temps d'une MÊME tranche épaisse (intérieure), on garde l'estimateur ordinaire, moyenné sur les N₁ échantillons.
//!
//! `d = 0` : aucun gel, toutes les paires « valides » ; exige N₁ = 1 (l'estimateur est alors exactement l'ordinaire).

/// Paramètres de l'option `--deux-niveaux D,N1[,S1]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamDeuxNiveaux {
    pub d: usize,
    pub n1: usize,
    /// balayages entre deux échantillons internes (défaut 1)
    pub s1: usize,
}

impl ParamDeuxNiveaux {
    pub fn analyser(t: &str, tt: usize) -> Result<ParamDeuxNiveaux, String> {
        let v: Vec<&str> = t.split(',').collect();
        if v.len() != 2 && v.len() != 3 {
            return Err(format!("--deux-niveaux « {t} » : forme D,N1[,S1] attendue"));
        }
        let s1: usize = if v.len() == 3 { v[2].trim().parse().map_err(|_| format!("S1 « {} » illisible", v[2]))? } else { 1 };
        if s1 == 0 {
            return Err("S1 ≥ 1 requis".into());
        }
        let d: usize = v[0].trim().parse().map_err(|_| format!("D « {} » illisible", v[0]))?;
        let n1: usize = v[1].trim().parse().map_err(|_| format!("N1 « {} » illisible", v[1]))?;
        if n1 == 0 {
            return Err("N1 ≥ 1 requis".into());
        }
        if d == 0 {
            if n1 != 1 {
                return Err("D = 0 (aucun gel) exige N1 = 1".into());
            }
        } else if d < 2 || !tt.is_multiple_of(d) || tt / d < 2 {
            return Err(format!("D = {d} : D ≥ 2, D | T = {tt} et au moins deux tranches épaisses requis"));
        }
        Ok(ParamDeuxNiveaux { d, n1, s1 })
    }
}

/// Masque de liens gelés : octet 1 pour les liens SPATIAUX (μ ≠ temps) des sites dont la coordonnée de temps ≡ 0 (mod d).
/// Disposition des liens : li = μ·V + s ; `coord_temps(s)` rend la coordonnée de temps du site s.
pub fn masque_gel(v: usize, temps: usize, d: usize, coord_temps: impl Fn(usize) -> usize) -> Vec<u8> {
    let mut m = vec![0u8; 4 * v];
    if d == 0 {
        return m;
    }
    for s in 0..v {
        if coord_temps(s).is_multiple_of(d) {
            for mu in 0..4 {
                if mu != temps {
                    m[mu * v + s] = 1;
                }
            }
        }
    }
    m
}

/// Tranche épaisse d'un temps intérieur ; `None` pour un temps gelé (constant à l'intérieur de la boucle interne).
pub fn tranche(t: usize, d: usize) -> Option<usize> {
    if d == 0 || t.is_multiple_of(d) {
        None
    } else {
        Some(t / d)
    }
}

/// La paire de temps (t, t') se factorise-t-elle (produit des moyennes internes) ?
pub fn paire_valide(t: usize, t2: usize, d: usize) -> bool {
    if d == 0 {
        return true;
    }
    match (tranche(t, d), tranche(t2, d)) {
        (Some(a), Some(b)) => a != b,
        _ => true,
    }
}

/// Nombre de paires de temps (t, t+τ mod T) valides, pour τ = 0..=tau_max.
pub fn paires_valides(tt: usize, d: usize, tau_max: usize) -> Vec<Vec<usize>> {
    (0..=tau_max).map(|tau| (0..tt).filter(|&t| paire_valide(t, (t + tau) % tt, d)).collect()).collect()
}

/// Accumulateur d'un jeu d'opérateurs (nops opérateurs, tt temps, `comp` composantes : 1 réel, 2 complexe (Re, Im)) pour UNE
/// chaîne : somme des O_a(t) sur les échantillons internes et somme des produits ordinaires sur les paires NON valides.
/// Produit de deux valeurs : Σ_k x_k y_k (= Re[x* y] pour comp = 2).
pub struct Cumul {
    nops: usize,
    tt: usize,
    comp: usize,
    tau_max: usize,
    d: usize,
    n: usize,
    somme: Vec<f64>,
    ordinaire: Vec<f64>,
}

impl Cumul {
    pub fn nouveau(nops: usize, tt: usize, tau_max: usize, d: usize) -> Cumul {
        Cumul::nouveau_comp(nops, tt, 1, tau_max, d)
    }

    pub fn nouveau_comp(nops: usize, tt: usize, comp: usize, tau_max: usize, d: usize) -> Cumul {
        let npair = nops * (nops + 1) / 2;
        Cumul { nops, tt, comp, tau_max, d, n: 0, somme: vec![0.0; nops * tt * comp], ordinaire: vec![0.0; npair * (tau_max + 1)] }
    }

    /// Cumul reconstruit à partir des sommes calculées sur la carte (`k_cumul_somme`, `k_cumul_ord`) :
    /// `somme[(a·tt + t)·comp + k]` (somme des O_a(t) sur n échantillons), `ordinaire[pair·(tau_max+1) + τ]` (somme des produits
    /// ordinaires sur les paires non valides).
    #[allow(clippy::too_many_arguments)] // style seulement (indices explicites conservés : aucun changement numérique)
    pub fn depuis_sommes(nops: usize, tt: usize, comp: usize, tau_max: usize, d: usize, n: usize, somme: Vec<f64>, ordinaire: Vec<f64>) -> Cumul {
        assert_eq!(somme.len(), nops * tt * comp);
        assert_eq!(ordinaire.len(), nops * (nops + 1) / 2 * (tau_max + 1));
        Cumul { nops, tt, comp, tau_max, d, n, somme, ordinaire }
    }

    fn prod(&self, o: &[f64], a: usize, t: usize, b: usize, t2: usize) -> f64 {
        let (ia, ib) = ((a * self.tt + t) * self.comp, (b * self.tt + t2) * self.comp);
        (0..self.comp).map(|k| o[ia + k] * o[ib + k]).sum()
    }

    /// Ajoute un échantillon interne : `o[(a·tt + t)·comp + k]`.
    pub fn ajouter(&mut self, o: &[f64]) {
        let (nops, tt) = (self.nops, self.tt);
        for (x, y) in self.somme.iter_mut().zip(o) {
            *x += y;
        }
        let mut pair = 0usize;
        for a in 0..nops {
            for b in a..nops {
                for tau in 0..=self.tau_max {
                    let mut acc = 0.0;
                    for t in 0..tt {
                        let t2 = (t + tau) % tt;
                        if !paire_valide(t, t2, self.d) {
                            acc += self.prod(o, a, t, b, t2) + self.prod(o, b, t, a, t2);
                        }
                    }
                    self.ordinaire[pair * (self.tau_max + 1) + tau] += acc;
                }
                pair += 1;
            }
        }
        self.n += 1;
    }

    /// Corrélateur [paire][τ] aplati (même ordre et même normalisation 1/(2T) que `k_corr`) et moyennes ⟨O_a⟩ (composante 0).
    pub fn corr(&self) -> (Vec<f64>, Vec<f64>) {
        let (nops, tt) = (self.nops, self.tt);
        let n = self.n.max(1) as f64;
        let moy_t: Vec<f64> = self.somme.iter().map(|x| x / n).collect();
        let mut out = Vec::with_capacity(self.ordinaire.len());
        let mut pair = 0usize;
        for a in 0..nops {
            for b in a..nops {
                for tau in 0..=self.tau_max {
                    let mut acc = self.ordinaire[pair * (self.tau_max + 1) + tau] / n;
                    for t in 0..tt {
                        let t2 = (t + tau) % tt;
                        if paire_valide(t, t2, self.d) {
                            acc += self.prod(&moy_t, a, t, b, t2) + self.prod(&moy_t, b, t, a, t2);
                        }
                    }
                    out.push(acc / (2.0 * tt as f64));
                }
                pair += 1;
            }
        }
        let moy: Vec<f64> = (0..nops)
            .map(|a| (0..tt).map(|t| moy_t[(a * tt + t) * self.comp]).sum::<f64>() / tt as f64)
            .collect();
        (out, moy)
    }
}

/// Cumul « brut » à toutes les impulsions : référence hôte du noyau `k_cumul_somme` / `k_cumul_ord` (mêmes opérations dans le même
/// ordre, donc mêmes bits en f64). Entrée : les sommes d'une chaîne `s[(a·tt + t)·nq + q]` (a = niveau·nf + forme, q comme dans
/// `sg`). `slot` −1 = impulsion nulle, k ≥ 0 = cran k (moyenne des 3 axes).
pub struct CumulBrut {
    nops: usize,
    tt: usize,
    nq: usize,
    tau_max: usize,
    d: usize,
    pub n: usize,
    pub somme: Vec<f64>,
    /// [slot+1][paire][τ]
    pub ordinaire: Vec<f64>,
}

impl CumulBrut {
    pub fn nouveau(nops: usize, tt: usize, nq: usize, tau_max: usize, d: usize) -> CumulBrut {
        let nslots = 1 + (nq - 1) / 6;
        let npair = nops * (nops + 1) / 2;
        CumulBrut { nops, tt, nq, tau_max, d, n: 0, somme: vec![0.0; nops * tt * nq], ordinaire: vec![0.0; nslots * npair * (tau_max + 1)] }
    }

    pub fn ajouter(&mut self, s: &[f64]) {
        let (nops, tt, nq, ntau) = (self.nops, self.tt, self.nq, self.tau_max + 1);
        let nslots = 1 + (nq - 1) / 6;
        for (x, y) in self.somme.iter_mut().zip(s) {
            *x += y;
        }
        let npair = nops * (nops + 1) / 2;
        let mut pair = 0usize;
        for a in 0..nops {
            for b in a..nops {
                let (ia, ib) = (a * tt * nq, b * tt * nq);
                for tau in 0..ntau {
                    for slot in -1..(nslots as i32 - 1) {
                        let mut acc = 0.0f64;
                        for t in 0..tt {
                            let t2 = (t + tau) % tt;
                            if paire_valide(t, t2, self.d) {
                                continue;
                            }
                            if slot < 0 {
                                acc += s[ia + t * nq] * s[ib + t2 * nq] + s[ib + t * nq] * s[ia + t2 * nq];
                            } else {
                                for dd in 0..3 {
                                    let q = 1 + slot as usize * 6 + dd * 2;
                                    let (ar, ai) = (s[ia + t * nq + q], s[ia + t * nq + q + 1]);
                                    let (br, bi) = (s[ib + t2 * nq + q], s[ib + t2 * nq + q + 1]);
                                    let (ar2, ai2) = (s[ib + t * nq + q], s[ib + t * nq + q + 1]);
                                    let (br2, bi2) = (s[ia + t2 * nq + q], s[ia + t2 * nq + q + 1]);
                                    acc += (ar * br + ai * bi) + (ar2 * br2 + ai2 * bi2);
                                }
                            }
                        }
                        self.ordinaire[((slot + 1) as usize * npair + pair) * ntau + tau] += acc;
                    }
                }
                pair += 1;
            }
        }
        self.n += 1;
    }
}

/// Corrélateur à l'impulsion `slot` (−1 : nulle ; k : cran k, moyenne des 3 axes, normalisation 1/(2T·3)) à partir des sommes
/// `somme[(a·tt + t)·nq + q]` et `ordinaire[(slot+1)·npair·ntau + pair·ntau + τ]` (n échantillons). Paires valides : produit des
/// moyennes internes ; non valides : somme des produits ordinaires moyennée.
#[allow(clippy::too_many_arguments)] // style seulement (indices explicites conservés : aucun changement numérique)
pub fn corr_somme(somme: &[f64], ordinaire: &[f64], n: usize, nops: usize, tt: usize, nq: usize, tau_max: usize, d: usize, slot: i32) -> Vec<f64> {
    let ntau = tau_max + 1;
    let npair = nops * (nops + 1) / 2;
    let nn = n.max(1) as f64;
    let m: Vec<f64> = somme.iter().map(|x| x / nn).collect();
    let mut out = Vec::with_capacity(npair * ntau);
    let mut pair = 0usize;
    for a in 0..nops {
        for b in a..nops {
            let (ia, ib) = (a * tt * nq, b * tt * nq);
            for tau in 0..ntau {
                let mut acc = ordinaire[((slot + 1) as usize * npair + pair) * ntau + tau] / nn;
                for t in 0..tt {
                    let t2 = (t + tau) % tt;
                    if !paire_valide(t, t2, d) {
                        continue;
                    }
                    if slot < 0 {
                        acc += m[ia + t * nq] * m[ib + t2 * nq] + m[ib + t * nq] * m[ia + t2 * nq];
                    } else {
                        for dd in 0..3 {
                            let q = 1 + slot as usize * 6 + dd * 2;
                            let (ar, ai) = (m[ia + t * nq + q], m[ia + t * nq + q + 1]);
                            let (br, bi) = (m[ib + t2 * nq + q], m[ib + t2 * nq + q + 1]);
                            let (ar2, ai2) = (m[ib + t * nq + q], m[ib + t * nq + q + 1]);
                            let (br2, bi2) = (m[ia + t2 * nq + q], m[ia + t2 * nq + q + 1]);
                            acc += (ar * br + ai * bi) + (ar2 * br2 + ai2 * bi2);
                        }
                    }
                }
                out.push(if slot < 0 { acc / (2.0 * tt as f64) } else { acc / (2.0 * tt as f64 * 3.0) });
            }
            pair += 1;
        }
    }
    out
}

/// Corrélateur ordinaire d'un échantillon unique (référence processeur de `k_corr`, impulsion nulle).
pub fn corr_ordinaire(o: &[f64], nops: usize, tt: usize, tau_max: usize) -> Vec<f64> {
    let mut c = Cumul::nouveau(nops, tt, tau_max, 0);
    c.ajouter(o);
    c.corr().0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyse_et_paires() {
        assert!(ParamDeuxNiveaux::analyser("4,10", 16).is_ok());
        assert!(ParamDeuxNiveaux::analyser("5,10", 16).is_err());
        assert!(ParamDeuxNiveaux::analyser("16,10", 16).is_err());
        assert!(ParamDeuxNiveaux::analyser("0,3", 16).is_err());
        assert!(ParamDeuxNiveaux::analyser("0,1", 16).is_ok());
        // d = 4 : t = 1 et t' = 2 dans la même tranche, t = 1 et 5 dans deux tranches, 0 gelé
        assert!(!paire_valide(1, 2, 4));
        assert!(paire_valide(1, 5, 4));
        assert!(paire_valide(0, 2, 4));
        assert!(paire_valide(3, 4, 4));
        let pv = paires_valides(16, 4, 3);
        assert_eq!(pv[0].len(), 4); // τ = 0 : seuls les temps gelés
        // τ = 1 : (0,1) (3,4) (4,5) (7,8) (8,9) (11,12) (12,13) (15,0) valides : 8 ; (1,2),(2,3),(5,6)… non
        assert_eq!(pv[1].len(), 8);
    }

    #[test]
    fn masque() {
        let m = masque_gel(32, 0, 4, |s| s / 4);
        // temps 0, v = 32 : sites 0..4 (coord 0) et 16..20 (coord 4) gelés pour μ = 1..3
        assert_eq!(m.iter().filter(|&&x| x == 1).count(), 3 * 8);
        assert_eq!(m[0], 0);
        assert_eq!(m[32], 1);
    }

    #[test]
    fn un_echantillon_sans_gel_est_ordinaire() {
        let (nops, tt) = (2, 8);
        let o: Vec<f64> = (0..nops * tt).map(|i| ((i * 7 + 3) % 11) as f64 - 5.0).collect();
        let a = corr_ordinaire(&o, nops, tt, 3);
        // N1 = 1 avec gel d = 4 : toute paire est soit factorisée (produit des moyennes = produit) soit ordinaire : même valeur
        let mut c = Cumul::nouveau(nops, tt, 3, 4);
        c.ajouter(&o);
        let b = c.corr().0;
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-12);
        }
    }
}
