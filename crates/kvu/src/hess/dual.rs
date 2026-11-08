//! DŪ·v par NOMBRES DUAUX sur la carte de Bałaban (WP13, indépendant de WP11/WP12a).
//!
//! La carte `Ū` est celle de `rg::bloc` (géométrie 3centre, 10 473 boucles par lien grossier) : même tables de chemins, même
//! ordre de sommation, mais recodée ici en arithmétique duale (valeur + K dérivées directionnelles) ; aucune formule fermée
//! de WP11 n'est utilisée. Convention : perturbation À GAUCHE `U_l → exp(i ε·σ) U_l` avec `ε = ζ/2` (coordonnées `a` de S2 :
//! `exp(i ζ·σ/2)`), et sortie exprimée dans les mêmes coordonnées : `ζ'_c = 2 vec(dŪ_c Ū_c†)`.
//! Les matrices de DŪ (3n × 3N) sont donc identiques dans toutes les bases (même mise à l'échelle des deux côtés).
//!
//! Disposition des vecteurs : « s2 » (lien `4x + μ`, indice `3 lien + a`), côté fin et côté grossier.

use super::bases::Reseau;
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::rg::geometrie::{Geometrie, Pas, Table, SITES_BLOC};
use rayon::prelude::*;

/// Nombre dual à K composantes de dérivée.
#[derive(Clone, Copy, Debug)]
pub struct Dual<const K: usize> {
    pub v: f64,
    pub d: [f64; K],
}

impl<const K: usize> Dual<K> {
    #[inline]
    pub fn cst(v: f64) -> Self {
        Dual { v, d: [0.0; K] }
    }
    #[inline]
    pub fn add(self, o: Self) -> Self {
        let mut d = [0.0; K];
        for k in 0..K {
            d[k] = self.d[k] + o.d[k];
        }
        Dual { v: self.v + o.v, d }
    }
    #[inline]
    pub fn sub(self, o: Self) -> Self {
        let mut d = [0.0; K];
        for k in 0..K {
            d[k] = self.d[k] - o.d[k];
        }
        Dual { v: self.v - o.v, d }
    }
    #[inline]
    pub fn mul(self, o: Self) -> Self {
        let mut d = [0.0; K];
        for k in 0..K {
            d[k] = self.d[k] * o.v + self.v * o.d[k];
        }
        Dual { v: self.v * o.v, d }
    }
    #[inline]
    pub fn neg(self) -> Self {
        let mut d = [0.0; K];
        for k in 0..K {
            d[k] = -self.d[k];
        }
        Dual { v: -self.v, d }
    }
    #[inline]
    pub fn echelle(self, s: f64) -> Self {
        let mut d = [0.0; K];
        for k in 0..K {
            d[k] = self.d[k] * s;
        }
        Dual { v: self.v * s, d }
    }
    /// f(x) de dérivée première `df` (valeur de la fonction et de sa dérivée en x.v).
    #[inline]
    fn compose(self, f: f64, df: f64) -> Self {
        let mut d = [0.0; K];
        for k in 0..K {
            d[k] = self.d[k] * df;
        }
        Dual { v: f, d }
    }
}

pub type Qd<const K: usize> = [Dual<K>; 4];

#[inline]
pub fn qmul<const K: usize>(a: &Qd<K>, b: &Qd<K>) -> Qd<K> {
    // même règle que su2_hb::qmul : a0 b0 − a·b ; a0 b + b0 a − a×b
    let m = |x: Dual<K>, y: Dual<K>| x.mul(y);
    [
        m(a[0], b[0]).sub(m(a[1], b[1])).sub(m(a[2], b[2])).sub(m(a[3], b[3])),
        m(a[0], b[1]).add(m(b[0], a[1])).sub(m(a[2], b[3]).sub(m(a[3], b[2]))),
        m(a[0], b[2]).add(m(b[0], a[2])).sub(m(a[3], b[1]).sub(m(a[1], b[3]))),
        m(a[0], b[3]).add(m(b[0], a[3])).sub(m(a[1], b[2]).sub(m(a[2], b[1]))),
    ]
}

#[inline]
pub fn qdag<const K: usize>(a: &Qd<K>) -> Qd<K> {
    [a[0], a[1].neg(), a[2].neg(), a[3].neg()]
}

fn identite<const K: usize>() -> Qd<K> {
    [Dual::cst(1.0), Dual::cst(0.0), Dual::cst(0.0), Dual::cst(0.0)]
}

/// Logarithme `log_q` de `rg::bloc` (vecteur ω = θ n), mêmes branches (série sous 1e-6).
pub fn log_q<const K: usize>(w: &Qd<K>) -> [Dual<K>; 3] {
    let n2 = w[1].mul(w[1]).add(w[2].mul(w[2])).add(w[3].mul(w[3]));
    let facteur = if n2.v.sqrt() < 1e-6 {
        let t = n2.mul(n2);
        Dual::cst(1.0).add(n2.echelle(1.0 / 6.0)).add(t.echelle(7.0 / 360.0))
    } else {
        let norme = n2.compose(n2.v.sqrt(), 0.5 / n2.v.sqrt());
        // θ = atan2(norme, w0) ; dθ = (w0 dnorme − norme dw0) / (norme² + w0²)
        let den = norme.v * norme.v + w[0].v * w[0].v;
        let theta_v = norme.v.atan2(w[0].v);
        let mut d = [0.0; K];
        for k in 0..K {
            d[k] = (w[0].v * norme.d[k] - norme.v * w[0].d[k]) / den;
        }
        let theta = Dual { v: theta_v, d };
        // θ / norme
        let inv = norme.compose(1.0 / norme.v, -1.0 / (norme.v * norme.v));
        theta.mul(inv)
    };
    [facteur.mul(w[1]), facteur.mul(w[2]), facteur.mul(w[3])]
}

/// `exp_q` de `rg::bloc` : exp(i s·σ), mêmes branches.
pub fn exp_q<const K: usize>(s: &[Dual<K>; 3]) -> Qd<K> {
    let n2 = s[0].mul(s[0]).add(s[1].mul(s[1])).add(s[2].mul(s[2]));
    let norme_v = n2.v.sqrt();
    if norme_v < 1e-6 {
        let t = n2.mul(n2);
        let facteur = Dual::cst(1.0).sub(n2.echelle(1.0 / 6.0)).add(t.echelle(1.0 / 120.0));
        // cos(norme) = 1 − n2/2 + n2²/24 (série, identique à 1e-24 près au cos exact à ce rayon)
        let c = Dual::cst(1.0).sub(n2.echelle(0.5)).add(t.echelle(1.0 / 24.0));
        [c, facteur.mul(s[0]), facteur.mul(s[1]), facteur.mul(s[2])]
    } else {
        let norme = n2.compose(norme_v, 0.5 / norme_v);
        let c = norme.compose(norme_v.cos(), -norme_v.sin());
        let sn = norme.compose(norme_v.sin(), norme_v.cos());
        let inv = norme.compose(1.0 / norme_v, -1.0 / (norme_v * norme_v));
        let facteur = sn.mul(inv);
        [c, facteur.mul(s[0]), facteur.mul(s[1]), facteur.mul(s[2])]
    }
}

/// Source de liens duaux : `lien(mu, site)`.
pub trait SourceLiens<const K: usize>: Sync {
    fn lien(&self, mu: usize, site: usize) -> Qd<K>;
}

fn produit<const K: usize, S: SourceLiens<K>>(
    src: &S,
    l: usize,
    origine: [usize; 4],
    chemin: &[Pas],
    translation: [i8; 4],
) -> Qd<K> {
    let mut p = identite::<K>();
    for pas in chemin {
        let x: [usize; 4] = std::array::from_fn(|m| {
            (origine[m] as isize + pas.dep[m] as isize + translation[m] as isize).rem_euclid(l as isize) as usize
        });
        let u = src.lien(pas.mu as usize, Reseau::site_de(x, l));
        p = qmul(&p, &if pas.signe > 0 { u } else { qdag(&u) });
    }
    p
}

/// Ū(z, μ) en arithmétique duale (même algorithme que `rg::bloc::Tampon::lien`).
pub fn ubar_dual<const K: usize, S: SourceLiens<K>>(src: &S, table: &Table, l: usize, z: [usize; 4], mu: usize) -> Qd<K> {
    let origine = z.map(|v| 3 * v);
    let mut translation = [0i8; 4];
    translation[mu] = 3;
    let depart: Vec<Qd<K>> = table.chemins.iter().map(|p| produit(src, l, origine, p, [0; 4])).collect();
    let retour: Vec<Qd<K>> = table.chemins.iter().map(|p| qdag(&produit(src, l, origine, p, translation))).collect();
    let segments: Vec<Qd<K>> = (0..SITES_BLOC).map(|ir| produit(src, l, origine, &table.segment(mu, ir), [0; 4])).collect();
    let u0 = segments[Table::ir_de(table.ref_site)];
    let inverse = qdag(&u0);
    let mut s = [Dual::<K>::cst(0.0); 3];
    for ir in 0..SITES_BLOC {
        let poids = table.poids(ir);
        let (debut, fin) = (table.debut[ir], table.debut[ir + 1]);
        for i in debut..fin {
            let pre = qmul(&qmul(&inverse, &depart[i]), &segments[ir]);
            for j in debut..fin {
                let ell = log_q(&qmul(&pre, &retour[j]));
                for m in 0..3 {
                    s[m] = s[m].add(ell[m].echelle(poids));
                }
            }
        }
    }
    qmul(&u0, &exp_q(&s))
}

// ------------------------------------------------------------------------------------------
// DŪ·v pour une direction (K = 1)
// ------------------------------------------------------------------------------------------

struct LiensUneDirection<'a> {
    champ: &'a ChampHote,
    /// dérivée de chaque lien (4 composantes par lien « champ »), nulle si v = 0.
    dq: Vec<[f64; 4]>,
}

impl SourceLiens<1> for LiensUneDirection<'_> {
    #[inline]
    fn lien(&self, mu: usize, site: usize) -> Qd<1> {
        let e = mu * self.champ.v() + site;
        let q = self.champ.lien(e);
        let d = self.dq[e];
        std::array::from_fn(|i| Dual { v: q[i], d: [d[i]] })
    }
}

fn mul_pur(eps: [f64; 3], q: [f64; 4]) -> [f64; 4] {
    // [0, ε] ⊗ q  (règle de qmul avec a0 = 0)
    [
        -(eps[0] * q[1] + eps[1] * q[2] + eps[2] * q[3]),
        eps[0] * q[0] - (eps[1] * q[3] - eps[2] * q[2]),
        eps[1] * q[0] - (eps[2] * q[1] - eps[0] * q[3]),
        eps[2] * q[0] - (eps[0] * q[2] - eps[1] * q[1]),
    ]
}

/// Sortie de [`jvp`] : Ū (quaternions, disposition `ChampHote` grossière : lien `μ Vc + z`) et ζ' (disposition s2, 3 par lien `4z + μ`).
pub struct SortieJvp {
    pub ubar: ChampHote,
    pub dubar_v: Vec<f64>,
}

fn verifier_champ(champ: &ChampHote) -> Resultat<()> {
    if champ.l < 3 || !champ.l.is_multiple_of(3) {
        return Err(Erreur::Config(format!("L = {} : multiple de 3 ≥ 3 requis pour la carte de Bałaban", champ.l)));
    }
    if champ.q.len() != 16 * champ.v() {
        return Err(Erreur::Config("longueur du champ invalide".into()));
    }
    Ok(())
}

/// DŪ·v : `v` en disposition s2 (3 · 4L⁴), résultat en disposition s2 grossière (3 · 4 (L/3)⁴). Indépendant du nombre de fils (liens grossiers indépendants).
pub fn jvp(pool: &rayon::ThreadPool, champ: &ChampHote, v: &[f64]) -> Resultat<SortieJvp> {
    verifier_champ(champ)?;
    let reseau = Reseau::nouveau(champ.l)?;
    if v.len() != reseau.n {
        return Err(Erreur::Config(format!("vecteur de longueur {} au lieu de {}", v.len(), reseau.n)));
    }
    let mut dq = vec![[0.0; 4]; reseau.nl];
    for l in 0..reseau.nl {
        let e = reseau.champ_de_s2(l);
        let eps = [0.5 * v[3 * l], 0.5 * v[3 * l + 1], 0.5 * v[3 * l + 2]];
        dq[e] = mul_pur(eps, champ.lien(e));
    }
    let src = LiensUneDirection { champ, dq };
    let table = Table::nouvelle(Geometrie::TroisCentre);
    let lc = champ.l / 3;
    let rc = Reseau::nouveau(lc.max(2)).ok();
    let vc = lc.pow(4);
    let ncoarse = 4 * vc;
    let l = champ.l;
    let res: Vec<(Qd<1>, usize)> = pool.install(|| {
        (0..ncoarse)
            .into_par_iter()
            .map(|e| {
                let mu = e / vc;
                let z = Reseau::coordonnees_de(e % vc, lc);
                (ubar_dual(&src, &table, l, z, mu), e)
            })
            .collect()
    });
    let _ = rc;
    let mut q = vec![0.0; 4 * ncoarse];
    let mut dv = vec![0.0; 3 * ncoarse];
    for (u, e) in res {
        let mu = e / vc;
        let zs = e % vc;
        let val = [u[0].v, u[1].v, u[2].v, u[3].v];
        q[4 * e..4 * e + 4].copy_from_slice(&val);
        let du = [u[0].d[0], u[1].d[0], u[2].d[0], u[3].d[0]];
        // dŪ Ū† : vecteur → ζ' = 2 vec
        let dag = [val[0], -val[1], -val[2], -val[3]];
        let p = [
            du[0] * dag[0] - du[1] * dag[1] - du[2] * dag[2] - du[3] * dag[3],
            du[0] * dag[1] + dag[0] * du[1] - (du[2] * dag[3] - du[3] * dag[2]),
            du[0] * dag[2] + dag[0] * du[2] - (du[3] * dag[1] - du[1] * dag[3]),
            du[0] * dag[3] + dag[0] * du[3] - (du[1] * dag[2] - du[2] * dag[1]),
        ];
        let ls = 4 * zs + mu;
        dv[3 * ls] = 2.0 * p[1];
        dv[3 * ls + 1] = 2.0 * p[2];
        dv[3 * ls + 2] = 2.0 * p[3];
    }
    Ok(SortieJvp { ubar: ChampHote { l: lc, q }, dubar_v: dv })
}

// ------------------------------------------------------------------------------------------
// Jacobien dense DŪ (lots de K directions par lien grossier)
// ------------------------------------------------------------------------------------------

const KB: usize = 24;

struct LiensLot<'a> {
    champ: &'a ChampHote,
    /// lien « champ » → indice local dans `locaux` (u32::MAX si hors support)
    index: &'a [u32],
    locaux: Vec<Qd<KB>>,
}

impl SourceLiens<KB> for LiensLot<'_> {
    #[inline]
    fn lien(&self, mu: usize, site: usize) -> Qd<KB> {
        let e = mu * self.champ.v() + site;
        let i = self.index[e];
        if i == u32::MAX {
            let q = self.champ.lien(e);
            std::array::from_fn(|c| Dual::cst(q[c]))
        } else {
            self.locaux[i as usize]
        }
    }
}

/// Liens « champ » touchés par la carte au lien grossier (z, μ) (union des chemins de départ, de retour et des segments).
pub fn support_lien(table: &Table, l: usize, z: [usize; 4], mu: usize) -> Vec<usize> {
    let v = l.pow(4);
    let origine = z.map(|c| 3 * c);
    let mut translation = [0i8; 4];
    translation[mu] = 3;
    let mut vus = std::collections::BTreeSet::new();
    let mut ajouter = |chemin: &[Pas], tr: [i8; 4]| {
        for pas in chemin {
            let x: [usize; 4] =
                std::array::from_fn(|m| (origine[m] as isize + pas.dep[m] as isize + tr[m] as isize).rem_euclid(l as isize) as usize);
            vus.insert(pas.mu as usize * v + Reseau::site_de(x, l));
        }
    };
    for p in &table.chemins {
        ajouter(p, [0; 4]);
        ajouter(p, translation);
    }
    for ir in 0..SITES_BLOC {
        ajouter(&table.segment(mu, ir), [0; 4]);
    }
    vus.into_iter().collect()
}

/// Jacobien DŪ dense : `3 · 4 (L/3)⁴` lignes (disposition s2 grossière, 3 par lien `4z + μ`) de longueur `12 L⁴`
/// (disposition s2 fine). Chaque ligne est un `Vec<f64>`. Les liens hors support ont des colonnes nulles.
pub fn jacobien_dense(pool: &rayon::ThreadPool, champ: &ChampHote) -> Resultat<Vec<Vec<f64>>> {
    verifier_champ(champ)?;
    let reseau = Reseau::nouveau(champ.l)?;
    let table = Table::nouvelle(Geometrie::TroisCentre);
    let lc = champ.l / 3;
    let vc = lc.pow(4);
    let ncoarse = 4 * vc;
    let n = reseau.n;
    let l = champ.l;
    let blocs: Vec<(usize, [Vec<f64>; 3])> = pool.install(|| {
        (0..ncoarse)
            .into_par_iter()
            .map(|e| {
                let mu = e / vc;
                let zs = e % vc;
                let z = Reseau::coordonnees_de(zs, lc);
                let support = support_lien(&table, l, z, mu);
                let mut index = vec![u32::MAX; reseau.nl];
                for (i, &s) in support.iter().enumerate() {
                    index[s] = i as u32;
                }
                let mut lignes = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
                // directions (lien du support, couleur a)
                let dirs: Vec<(usize, usize)> = support.iter().flat_map(|&s| (0..3).map(move |a| (s, a))).collect();
                for lot in dirs.chunks(KB) {
                    let mut locaux: Vec<Qd<KB>> = support
                        .iter()
                        .map(|&s| {
                            let q = champ.lien(s);
                            std::array::from_fn(|c| Dual::cst(q[c]))
                        })
                        .collect();
                    for (k, &(s, a)) in lot.iter().enumerate() {
                        let q = champ.lien(s);
                        let mut eps = [0.0; 3];
                        eps[a] = 0.5;
                        let dq = mul_pur(eps, q);
                        let i = index[s] as usize;
                        for c in 0..4 {
                            locaux[i][c].d[k] = dq[c];
                        }
                    }
                    let src = LiensLot { champ, index: &index, locaux };
                    let u = ubar_dual(&src, &table, l, z, mu);
                    let val = [u[0].v, u[1].v, u[2].v, u[3].v];
                    let dag = [val[0], -val[1], -val[2], -val[3]];
                    for (k, &(s, a)) in lot.iter().enumerate() {
                        let du = [u[0].d[k], u[1].d[k], u[2].d[k], u[3].d[k]];
                        let p = [
                            du[0] * dag[0] - du[1] * dag[1] - du[2] * dag[2] - du[3] * dag[3],
                            du[0] * dag[1] + dag[0] * du[1] - (du[2] * dag[3] - du[3] * dag[2]),
                            du[0] * dag[2] + dag[0] * du[2] - (du[3] * dag[1] - du[1] * dag[3]),
                            du[0] * dag[3] + dag[0] * du[3] - (du[1] * dag[2] - du[2] * dag[1]),
                        ];
                        let lfin = reseau.s2_de_champ(s);
                        for b in 0..3 {
                            lignes[b][3 * lfin + a] = 2.0 * p[1 + b];
                        }
                    }
                }
                let ls = 4 * zs + mu;
                (ls, lignes)
            })
            .collect()
    });
    let mut j = vec![Vec::new(); 3 * ncoarse];
    for (ls, lignes) in blocs {
        let [a, b, c] = lignes;
        j[3 * ls] = a;
        j[3 * ls + 1] = b;
        j[3 * ls + 2] = c;
    }
    Ok(j)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_log_exp_derivees() {
        // d/dt exp(log(q(t))) = dq pour un quaternion unitaire q(t) = exp(i (s + t d)·σ)
        let s = [0.3, -0.2, 0.5];
        let d = [0.7, 0.1, -0.4];
        let sd: [Dual<1>; 3] = std::array::from_fn(|i| Dual { v: s[i], d: [d[i]] });
        let q = exp_q(&sd);
        let w = log_q(&q);
        for i in 0..3 {
            assert!((w[i].v - s[i]).abs() < 1e-14);
            assert!((w[i].d[0] - d[i]).abs() < 1e-12, "{} {}", w[i].d[0], d[i]);
        }
        // différences finies de exp
        let h = 1e-6;
        let qp = exp_q::<1>(&std::array::from_fn(|i| Dual::cst(s[i] + h * d[i])));
        let qm = exp_q::<1>(&std::array::from_fn(|i| Dual::cst(s[i] - h * d[i])));
        for c in 0..4 {
            assert!((q[c].d[0] - (qp[c].v - qm[c].v) / (2.0 * h)).abs() < 1e-8);
        }
    }

    #[test]
    fn dual_petites_normes() {
        let sd: [Dual<1>; 3] = std::array::from_fn(|i| Dual { v: 1e-9 * (i as f64 + 1.0), d: [1.0] });
        let q = exp_q(&sd);
        let w = log_q(&q);
        assert!(w.iter().all(|x| x.d[0].is_finite()));
        let sz: [Dual<1>; 3] = std::array::from_fn(|_| Dual { v: 0.0, d: [1.0] });
        let qz = exp_q(&sz);
        assert!(qz.iter().all(|x| x.d[0].is_finite()));
        let wz = log_q(&qz);
        assert!(wz.iter().all(|x| x.d[0].is_finite()));
    }
}
