//! Poids de Fourier Wf (= `cb.fourier_weights_sparse`) en CSR par ligne.
//!
//! Les 54 lignes de Wf (6 plans × 9 fonctionnelles pour pmax² = 1) sont rangées plan par plan, puis
//! par moment, puis (cos, sin) ; chaque valeur est f(ph)/L² avec ph = 2π (x·n) / L, et les zéros
//! EXACTS sont exclus (comme `np.flatnonzero`). L'ordre et les arrondis suivent le code Python.

use rayon::prelude::*;

/// Poids de Fourier Wf (nF × 6V) en CSR par ligne (= cb.fourier_weights_sparse(L, pmax2)).
#[derive(Clone, Debug, PartialEq)]
pub struct PoidsFourier {
    pub nf: usize,
    pub ptr: Vec<u64>,
    pub idx: Vec<u32>,
    pub val: Vec<f64>,
}

/// Fonction de la ligne : cosinus, ou sinus pour les moments non nuls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fonction {
    Cos,
    Sin,
}

/// Liste des moments n ∈ {-1, 0, 1}⁴ retenus, dans l'ordre de `fourier_weights_sparse` :
/// produit cartésien lexicographique, condition `0 < |n|² ≤ pmax2` ou n = 0, puis écartement de −n si n
/// a déjà été vu. (Le code Python n'enregistre que n dans `seen`, donc c'est le premier de la paire dans
/// l'ordre lexicographique, soit le moment à première composante non nulle négative, qui reste.)
fn moments(pmax2: u32) -> Vec<[i32; 4]> {
    let mut vus: Vec<[i32; 4]> = Vec::new();
    let mut moms: Vec<[i32; 4]> = Vec::new();
    for n0 in -1..=1 {
        for n1 in -1..=1 {
            for n2 in -1..=1 {
                for n3 in -1..=1 {
                    let n = [n0, n1, n2, n3];
                    let n2abs: i32 = n.iter().map(|x| x * x).sum();
                    let nabs: i32 = n.iter().map(|x| x.abs()).sum();
                    let retenu = (n2abs > 0 && n2abs as u32 <= pmax2) || nabs == 0;
                    if !retenu {
                        continue;
                    }
                    let moins = [-n0, -n1, -n2, -n3];
                    if vus.contains(&moins) {
                        continue;
                    }
                    vus.push(n);
                    moms.push(n);
                }
            }
        }
    }
    moms
}

/// Coordonnées (x0..x3) du site s (ordre C : s = ((x0 L + x1) L + x2) L + x3).
#[inline]
fn coords(l: usize, s: usize) -> [i64; 4] {
    [
        (s / (l * l * l)) as i64,
        ((s / (l * l)) % l) as i64,
        ((s / l) % l) as i64,
        (s % l) as i64,
    ]
}

/// Une ligne de Wf : (colonnes, valeurs) des entrées non nulles pour le plan `plan`, le moment `n`
/// et la fonction `f`.
fn ligne(l: usize, plan: usize, n: [i32; 4], f: Fonction) -> (Vec<u32>, Vec<f64>) {
    let v = l * l * l * l;
    let deux_pi = 2.0 * std::f64::consts::PI;
    let l2 = (l * l) as f64;
    let mut idx = Vec::new();
    let mut val = Vec::new();
    for s in 0..v {
        let x = coords(l, s);
        let k: i64 = (0..4).map(|d| x[d] * n[d] as i64).sum();
        // ph = ((2π) · k) / L : mêmes opérations que `2 * np.pi * (X @ n) / L`.
        let ph = deux_pi * (k as f64) / (l as f64);
        let w = match f {
            Fonction::Cos => ph.cos(),
            Fonction::Sin => ph.sin(),
        } / l2;
        if w != 0.0 {
            idx.push((plan * v + s) as u32);
            val.push(w);
        }
    }
    (idx, val)
}

/// Poids de Fourier de L⁴ pour `pmax2` (= `cb.fourier_weights_sparse(L, pmax2)` ; `pmax2` = 1 donne
/// les 54 fonctionnelles de la chaîne C-b).
///
/// # Panics
/// Si L est impair, nul, ou si 6 L⁴ ne tient pas dans un `u32`.
pub fn poids_fourier(l: usize, pmax2: u32) -> PoidsFourier {
    assert!(
        l >= 2 && l.is_multiple_of(2),
        "poids_fourier : L pair ≥ 2 requis (L = {l})"
    );
    let v = l * l * l * l;
    assert!(
        6 * v <= u32::MAX as usize,
        "poids_fourier : 6 L⁴ ne tient pas dans un u32"
    );
    let moms = moments(pmax2);
    let mut specs: Vec<(usize, [i32; 4], Fonction)> = Vec::new();
    for plan in 0..6 {
        for &n in &moms {
            specs.push((plan, n, Fonction::Cos));
            if n.iter().any(|&c| c != 0) {
                specs.push((plan, n, Fonction::Sin));
            }
        }
    }
    let lignes: Vec<(Vec<u32>, Vec<f64>)> = specs.par_iter().map(|&(plan, n, f)| ligne(l, plan, n, f)).collect();
    let nf = lignes.len();
    let mut ptr = Vec::with_capacity(nf + 1);
    ptr.push(0u64);
    let mut idx = Vec::new();
    let mut val = Vec::new();
    for (i, w) in lignes {
        idx.extend_from_slice(&i);
        val.extend_from_slice(&w);
        ptr.push(idx.len() as u64);
    }
    PoidsFourier { nf, ptr, idx, val }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moments_pmax2_1_comme_python() {
        // Python : les quatre moments à une composante −1, puis le moment nul.
        assert_eq!(
            moments(1),
            vec![[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1], [0, 0, 0, 0]]
        );
    }

    #[test]
    fn nombre_de_lignes_54() {
        let w = poids_fourier(4, 1);
        assert_eq!(w.nf, 54);
        assert_eq!(w.ptr.len(), 55);
        assert_eq!(*w.ptr.last().unwrap() as usize, w.idx.len());
        assert_eq!(w.idx.len(), w.val.len());
    }

    #[test]
    fn lignes_croissantes_et_sans_zero() {
        let w = poids_fourier(4, 1);
        for i in 0..w.nf {
            let (a, b) = (w.ptr[i] as usize, w.ptr[i + 1] as usize);
            assert!(b > a, "ligne vide");
            assert!(w.idx[a..b].windows(2).all(|p| p[0] < p[1]), "colonnes non croissantes");
            assert!(w.val[a..b].iter().all(|&x| x != 0.0));
        }
    }

    #[test]
    fn ligne_du_moment_nul_est_constante() {
        // Moment nul, cosinus : 1/L² partout (plan 0, ligne 8 : n = 0 est le 5ᵉ moment, 4 × 2 lignes avant).
        let l = 4;
        let w = poids_fourier(l, 1);
        let (a, b) = (w.ptr[8] as usize, w.ptr[9] as usize);
        assert_eq!(b - a, l * l * l * l);
        assert!(w.val[a..b].iter().all(|&x| x == 1.0 / 16.0));
    }

    #[test]
    fn colonnes_dans_le_plan_de_la_ligne() {
        let l = 4;
        let v = l * l * l * l;
        let w = poids_fourier(l, 1);
        for i in 0..w.nf {
            let plan = i / 9;
            for q in w.ptr[i] as usize..w.ptr[i + 1] as usize {
                assert_eq!(w.idx[q] as usize / v, plan);
            }
        }
    }

    #[test]
    fn sinus_exclut_les_zeros_exacts() {
        // Ligne (plan 0, n = (−1,0,0,0), sin) : ph = −2π x0 / L, nul exactement pour x0 = 0 (¼ des sites à L = 4).
        let l = 4;
        let w = poids_fourier(l, 1);
        let (a, b) = (w.ptr[1] as usize, w.ptr[2] as usize);
        assert_eq!(b - a, 3 * l * l * l, "x0 = 0 doit être exclu");
    }
}
