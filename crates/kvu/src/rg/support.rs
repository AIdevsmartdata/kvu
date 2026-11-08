//! Supports EXACTS des liens grossiers (WP22, `docs/BLOCAGE_QUASI_LOCAL.md` §5).
//!
//! Distance d_P entre liens : graphe des 4L⁴ liens du tore où deux liens distincts sont voisins ssi ils appartiennent à une même
//! plaquette (18 voisins par lien pour L ≥ 3 : exactement les liens des 6 agrafes du lien). Chemin droit de V(X, μ) : les liens
//! (3X + kμ̂, μ), k = 0, 1, 2. Supports : `dec` = chemin droit ; `ape:α:n` = boule d_P de rayon n ; `flot` (N pas RK3) = boule de
//! rayon 3N ; `agr` = chemin droit + liens des 6 agrafes de longueur 3 (33 liens) ; `3centre` = liens lus par `rg::bloc` (table S1).
//! Indices de lien : li = μ·V + s (rangement de `ChampHote`).

use super::bloc::{coordonnees, site};
use super::geometrie::{Geometrie, Table, SITES_BLOC};
use super::operateur::Operateur;
use std::collections::VecDeque;

fn deplace(x: [usize; 4], d: usize, delta: isize, l: usize) -> [usize; 4] {
    let mut y = x;
    y[d] = (x[d] as isize + delta).rem_euclid(l as isize) as usize;
    y
}

fn indice(l: usize, x: [usize; 4], mu: usize) -> usize {
    mu * l.pow(4) + site(x, l)
}

/// Les 18 liens qui partagent une plaquette avec `li` ; pour chaque ν ≠ μ croissant : (x, ν), (x+ν̂, μ), (x+μ̂, ν), (x−ν̂, ν),
/// (x−ν̂, μ), (x−ν̂+μ̂, ν).
pub fn voisins(l: usize, li: usize) -> [usize; 18] {
    let v = l.pow(4);
    let (mu, s) = (li / v, li % v);
    let x = coordonnees(s, l);
    let mut out = [0usize; 18];
    let mut k = 0;
    for nu in 0..4 {
        if nu == mu {
            continue;
        }
        let xm = deplace(x, nu, -1, l);
        for (y, d) in [
            (x, nu),
            (deplace(x, nu, 1, l), mu),
            (deplace(x, mu, 1, l), nu),
            (xm, nu),
            (xm, mu),
            (deplace(xm, mu, 1, l), nu),
        ] {
            out[k] = indice(l, y, d);
            k += 1;
        }
    }
    out
}

/// Chemin droit de V(z, μ) : (3z + kμ̂, μ), k = 0, 1, 2.
pub fn chemin_droit(l: usize, z: [usize; 4], mu: usize) -> [usize; 3] {
    let x = z.map(|t| (3 * t) % l);
    std::array::from_fn(|k| indice(l, deplace(x, mu, k as isize, l), mu))
}

/// Distances d_P depuis l'ensemble `depart` (parcours en largeur), jusqu'à `rmax` inclus ; `u32::MAX` au-delà.
pub fn distances(l: usize, depart: &[usize], rmax: usize) -> Vec<u32> {
    let n = 4 * l.pow(4);
    let mut d = vec![u32::MAX; n];
    let mut file = VecDeque::new();
    for &li in depart {
        if d[li] != 0 {
            d[li] = 0;
            file.push_back(li);
        }
    }
    while let Some(li) = file.pop_front() {
        let dl = d[li];
        if dl as usize >= rmax {
            continue;
        }
        for w in voisins(l, li) {
            if d[w] == u32::MAX {
                d[w] = dl + 1;
                file.push_back(w);
            }
        }
    }
    d
}

/// Liens lus par `rg::bloc` pour le lien grossier (z, μ) de `3centre` : chemins de la table (origine 3z, translations 0 et 3e_μ)
/// et les 81 segments, exactement aux adresses de `bloc.rs::produit`.
fn support_trois_centre(l: usize, z: [usize; 4], mu: usize) -> Vec<usize> {
    let table = Table::nouvelle(Geometrie::TroisCentre);
    let origine = z.map(|t| 3 * t);
    let mut liens = Vec::new();
    let mut lire = |dep: [i8; 4], d: u8, translation: [isize; 4]| {
        let x: [usize; 4] =
            std::array::from_fn(|m| (origine[m] as isize + dep[m] as isize + translation[m]).rem_euclid(l as isize) as usize);
        liens.push(indice(l, x, d as usize));
    };
    let mut tr = [0isize; 4];
    tr[mu] = 3;
    for p in &table.chemins {
        for pas in p {
            lire(pas.dep, pas.mu, [0; 4]);
            lire(pas.dep, pas.mu, tr);
        }
    }
    for ir in 0..SITES_BLOC {
        for pas in table.segment(mu, ir) {
            lire(pas.dep, pas.mu, [0; 4]);
        }
    }
    liens
}

/// Support exact (indices de liens fins TRIÉS, sans doublon) du lien grossier (z, μ) de l'opérateur `op` sur le tore L⁴.
pub fn support(op: &Operateur, l: usize, z: [usize; 4], mu: usize) -> Vec<usize> {
    let mut liens: Vec<usize> = match *op {
        Operateur::TroisCentre => support_trois_centre(l, z, mu),
        Operateur::Agr { .. } => {
            let x = z.map(|t| (3 * t) % l);
            let mut v: Vec<usize> = chemin_droit(l, z, mu).to_vec();
            for nu in 0..4 {
                if nu == mu {
                    continue;
                }
                for sg in [1isize, -1] {
                    let base = if sg == 1 { x } else { deplace(x, nu, -1, l) };
                    v.push(indice(l, base, nu));
                    v.push(indice(l, deplace(base, mu, 3, l), nu));
                    let xs = deplace(x, nu, sg, l);
                    for k in 0..3 {
                        v.push(indice(l, deplace(xs, mu, k, l), mu));
                    }
                }
            }
            v
        }
        _ => {
            let r = op.rayon_support().expect("rayon de support des opérateurs à boule");
            let d = distances(l, &chemin_droit(l, z, mu), r);
            d.iter().enumerate().filter(|(_, &x)| (x as usize) <= r).map(|(i, _)| i).collect()
        }
    };
    liens.sort_unstable();
    liens.dedup();
    liens
}

/// Texte de définition du support (manifestes).
pub fn definition(op: &Operateur) -> String {
    match *op {
        Operateur::TroisCentre => "liens des chemins de la table S1 (Γ, segments, Γ′) lus par le bloc de Bałaban".into(),
        Operateur::Dec => "chemin droit : (3X + kμ̂, μ), k = 0, 1, 2".into(),
        Operateur::Ape { n, .. } => format!("boule de rayon d_P = {n} (n pas APE) autour du chemin droit"),
        Operateur::Flot { n, .. } => format!("boule de rayon d_P = {} (3 × {n} pas RK3) autour du chemin droit", 3 * n),
        Operateur::Agr { .. } => "chemin droit + liens des 6 agrafes de longueur 3 (33 liens)".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(s: &str) -> Operateur {
        Operateur::analyser(s).unwrap()
    }

    #[test]
    fn voisins_symetriques_et_distincts() {
        for l in [3usize, 6] {
            for li in 0..4 * l.pow(4) {
                let v = voisins(l, li);
                let mut w = v.to_vec();
                w.sort_unstable();
                w.dedup();
                assert_eq!(w.len(), 18, "L = {l}, lien {li}");
                assert!(!w.contains(&li));
                for x in v {
                    assert!(voisins(l, x).contains(&li), "symétrie L = {l}, {li} ↔ {x}");
                }
            }
        }
    }

    /// Tailles contre le calcul indépendant `validation/rg_ql/support_py.py` (consignées dans la doc §5 avant ce code).
    #[test]
    fn tailles_contre_python() {
        let d = distances(12, &chemin_droit(12, [0; 4], 0), 30);
        let boule = |r: u32| d.iter().filter(|&&x| x <= r).count();
        for (r, n) in [(0, 3), (1, 45), (2, 245), (3, 793), (6, 7377), (7, 12281), (12, 53973), (13, 62129), (14, 68945)] {
            assert_eq!(boule(r), n, "L = 12, R = {r}");
        }
        assert_eq!(*d.iter().max().unwrap(), 23, "saturation du tore 12⁴ à R = 23");
        let d9 = distances(9, &chemin_droit(9, [0; 4], 0), 30);
        for (r, n) in [(6, 6837), (7, 10477), (13, 25852), (14, 26144)] {
            assert_eq!(d9.iter().filter(|&&x| x <= r).count(), n, "L = 9, R = {r}");
        }
        assert_eq!(support(&op("agr:1"), 12, [0; 4], 0).len(), 33);
        assert_eq!(support(&op("ape:0.5:13"), 12, [0; 4], 0).len(), 62129);
        assert_eq!(support(&op("flot:0.02:0.01"), 12, [0; 4], 0).len(), 7377);
        assert_eq!(support(&op("dec"), 12, [1, 2, 0, 1], 3).len(), 3);
        for mu in 1..4 {
            assert_eq!(support(&op("ape:0.5:6"), 12, [0; 4], mu).len(), 7377);
            assert_eq!(support(&op("agr:1"), 12, [1, 2, 0, 1], mu).len(), 33);
        }
    }

    #[test]
    fn support_trois_centre_dans_deux_blocs() {
        let s = support(&op("3centre"), 12, [1, 1, 1, 1], 2);
        assert!(!s.is_empty());
        // tous les liens lus ont leur base dans les blocs z et z + μ̂ élargis d'un site (chemins du centre aux sites du bloc)
        for li in &s {
            let x = coordonnees(li % 12usize.pow(4), 12);
            for (m, &xm) in x.iter().enumerate() {
                let lo = 3usize;
                let hi = if m == 2 { 9 } else { 6 };
                assert!(xm >= lo && xm < hi, "lien {li} hors des blocs : {x:?}");
            }
        }
    }
}
