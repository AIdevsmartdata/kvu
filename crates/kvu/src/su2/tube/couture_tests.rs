//! Portes de qualification de la couture DÉCALÉE (M7, tores obliques) : `GeoA::avec_couture`, `su2 gen --cotes --decalage`.
//!
//! Portes EXACTES (aucune statistique) :
//! 1. décalage nul ou période entière = réseau ordinaire (égalité de structure) et chaîne identique AU BIT ;
//! 2. combinatoire : voisins haut/bas inverses, L tours de l'axe recollé = translation par le décalage, plaquettes fermées, coloriage propre ;
//! 3. changement de base unimodulaire : deux descriptions du MÊME sous-réseau Γ (tore K de D-1) donnent un isomorphisme de graphes ;
//! 4. champs SU(2) aléatoires : action et agrafes invariantes sous cet isomorphisme ; invariance de jauge (plaquettes fermées) ;
//! 5. jauge ℤ₂ 3D et Ising 2D exacts (énumération) : valeur de contrôle D-1 (0,00139478169078), Z_K ≤ min(Z_A, Z_B),
//!    log-convexité des tores tournés, 0 ≤ a_d ≤ a_K ≤ a₃, invariance de Z sous le changement de base.

use super::reseau::{action, plaquette, ChaineTube};
use crate::geom::reseau::{GeoA, Pile};
use crate::geom::torsion::Torsion;
use crate::heritage::su2_hb::{qdag, qmul, qnorm, qscale, Q};

/// Réseau avec côtés ≥ 1 (les côtés égaux à 1 ne servent qu'aux tests exacts en dimension 2 ou 3) et couture éventuelle.
fn geo_libre(d: [usize; 4], axe: usize, vec: [i64; 4]) -> GeoA {
    use crate::geom::reseau::Couture;
    let v: usize = d.iter().product();
    let mut r = [0usize; 4];
    for i in 0..4 {
        r[i] = vec[i].rem_euclid(d[i] as i64) as usize;
    }
    assert_eq!(r[axe], 0);
    GeoA {
        d,
        v,
        st: [d[1] * d[2] * d[3], d[2] * d[3], d[3], 1],
        couture: if r.iter().any(|&x| x != 0) { Some(Couture { axe, vec: r }) } else { None },
    }
}

fn haut(g: &GeoA, s: usize, mu: usize) -> usize {
    g.nb_up(s, mu, g.coords(s)[mu])
}
fn bas(g: &GeoA, s: usize, mu: usize) -> usize {
    g.nb_dn(s, mu, g.coords(s)[mu])
}

// ------------------------------------------------------------------ porte 1
#[test]
fn decalage_nul_ou_periode_entiere_est_le_reseau_ordinaire() {
    let d = [4, 6, 3, 2];
    let ref_ = GeoA::nouvelle(d).unwrap();
    assert_eq!(GeoA::avec_couture(d, 0, [0, 0, 0, 0]).unwrap(), ref_);
    assert_eq!(GeoA::avec_couture(d, 0, [0, 6, 3, -2]).unwrap(), ref_); // périodes entières
    assert_eq!(GeoA::avec_couture(d, 1, [8, 0, -3, 2]).unwrap(), ref_);
    assert!(GeoA::avec_couture(d, 0, [1, 0, 0, 0]).is_err()); // composante selon l'axe recollé
    assert!(GeoA::avec_couture(d, 4, [0; 4]).is_err());
    assert!(GeoA::avec_couture(d, 0, [0, 1, 0, 0]).unwrap().couture.is_some());
}

#[test]
fn chaine_decalage_nul_identique_au_bit() {
    for d in [[4, 4, 4, 4], [3, 4, 2, 4], [5, 4, 4, 4]] {
        let a = GeoA::nouvelle(d).unwrap();
        let b = GeoA::avec_couture(d, 0, [0, d[1] as i64, 0, 0]).unwrap(); // période entière
        let pile = Pile::depuis_torsion(&a, &Torsion::triviale());
        let (mut ca, mut cb) = (ChaineTube::nouvelle(a, 7), ChaineTube::nouvelle(b, 7));
        ca.init(true);
        cb.init(true);
        for _ in 0..4 {
            ca.sweep_hb(2.1, &pile);
            cb.sweep_hb(2.1, &pile);
            ca.sweep_or(&pile);
            cb.sweep_or(&pile);
        }
        assert!(ca.u.iter().zip(&cb.u).all(|(x, y)| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())), "{d:?}");
        assert_eq!(ca.mesure5(&pile).map(f64::to_bits), cb.mesure5(&pile).map(f64::to_bits));
    }
}

// ------------------------------------------------------------------ porte 2
#[test]
fn combinatoire_couture() {
    let cas: [([usize; 4], usize, [i64; 4]); 6] = [
        ([3, 12, 2, 2], 0, [0, -3, 0, 0]),
        ([6, 12, 2, 2], 0, [0, -6, 1, 0]),
        ([4, 6, 3, 2], 1, [2, 0, 1, 1]),
        ([5, 4, 4, 3], 3, [1, 2, 3, 0]),
        ([2, 2, 2, 2], 2, [1, 1, 0, 1]),
        ([3, 4, 5, 2], 0, [0, 1, 4, 1]),
    ];
    for (d, axe, vec) in cas {
        let g = GeoA::avec_couture(d, axe, vec).unwrap();
        let c = g.couture.clone().unwrap();
        for s in 0..g.v {
            for mu in 0..4 {
                assert_eq!(bas(&g, haut(&g, s, mu), mu), s, "{d:?} {axe} {vec:?}");
                assert_eq!(haut(&g, bas(&g, s, mu), mu), s);
                for nu in 0..4 {
                    if nu != mu {
                        assert_eq!(haut(&g, haut(&g, s, mu), nu), haut(&g, haut(&g, s, nu), mu), "plaquette ouverte");
                    }
                }
            }
            // L tours de l'axe recollé = translation par `vec` (les autres axes : identité)
            let tour = g.decaler(s, axe, d[axe]);
            let (x, y) = (g.coords(s), g.coords(tour));
            for i in 0..4 {
                assert_eq!(y[i], (x[i] + c.vec[i]) % d[i]);
            }
            // tour d'un autre axe : identité
            for mu in (0..4).filter(|&m| m != axe) {
                assert_eq!(g.decaler(s, mu, d[mu]), s);
            }
        }
        let cl = g.couleurs();
        g.verifier_classes(&cl).unwrap();
    }
}

// ------------------------------------------------------------------ porte 3
/// Représentant du site de coordonnées entières `p` (le point est ramené dans le domaine fondamental de `g`).
#[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
fn reduire(g: &GeoA, p: [i64; 4]) -> usize {
    let mut p = p;
    if let Some(c) = &g.couture {
        let l = g.d[c.axe] as i64;
        let k = p[c.axe].div_euclid(l);
        p[c.axe] -= k * l;
        for i in 0..4 {
            p[i] -= k * c.vec[i] as i64;
        }
    }
    let mut x = [0usize; 4];
    for i in 0..4 {
        x[i] = p[i].rem_euclid(g.d[i] as i64) as usize;
    }
    g.site(&x)
}

/// Isomorphisme de graphes φ : g1 → g2 (même sous-réseau Γ, même ordre des axes), vérifié sur les 8 voisins de chaque site.
fn isomorphisme(g1: &GeoA, g2: &GeoA) -> Vec<usize> {
    assert_eq!(g1.v, g2.v);
    let phi: Vec<usize> = (0..g1.v).map(|s| reduire(g2, g1.coords(s).map(|x| x as i64))).collect();
    let mut vu = vec![false; g1.v];
    for &t in &phi {
        assert!(!vu[t], "φ non injective");
        vu[t] = true;
    }
    for s in 0..g1.v {
        for mu in 0..4 {
            assert_eq!(phi[haut(g1, s, mu)], haut(g2, phi[s], mu), "voisin haut, axe {mu}");
            assert_eq!(phi[bas(g1, s, mu)], bas(g2, phi[s], mu), "voisin bas, axe {mu}");
        }
    }
    phi
}

/// Les deux descriptions du tore K de D-1 : ⟨(2a,0),(a,b)⟩ = ⟨(0,2b),(a,b)⟩ (plan (x, t) ; axes ordonnés t, x, y, 1).
fn deux_k(a: usize, b: usize, ly: usize) -> (GeoA, GeoA) {
    // temps de longueur b recollé avec décalage a selon x (longueur 2a)  /  x de longueur a recollé avec décalage b selon t (longueur 2b)
    (geo_libre([b, 2 * a, ly, 1], 0, [0, a as i64, 0, 0]), geo_libre([2 * b, a, ly, 1], 1, [b as i64, 0, 0, 0]))
}

#[test]
fn changement_de_base_unimodulaire_isomorphisme() {
    for (a, b, ly) in [(2, 3, 2), (3, 2, 2), (2, 2, 3), (4, 3, 2), (1, 2, 2), (2, 1, 2)] {
        let (g1, g2) = deux_k(a, b, ly);
        isomorphisme(&g1, &g2);
        isomorphisme(&g2, &g1);
    }
    // cas M2 : (−3 ; 3) sur 12 × 8 × 8 = ⟨(12,0),(−3,3)⟩ ; même tore décrit par x recollé (longueur 3 sur l'axe) : (−3,3) → x : 3 sites, t : 6
    let g1 = GeoA::avec_couture([3, 12, 2, 2], 0, [0, -3, 0, 0]).unwrap();
    let g2 = GeoA::avec_couture([6, 3, 2, 2], 1, [3, 0, 0, 0]).unwrap();
    // ⟨(12,0),(−3,3)⟩ contient (−3,3)+(12,0)·k et (3,3)·... : (3,3)? non : (−3,3)+(6,0)? 6 ∉ Γ_x seul ; on vérifie seulement ce qui est vrai
    // (la réduction ci-dessus est un isomorphisme si et seulement si les deux Γ coïncident) : ici Γ₁ ≠ Γ₂, la fonction doit refuser.
    let r = std::panic::catch_unwind(|| isomorphisme(&g1, &g2));
    assert!(r.is_err(), "deux sous-réseaux différents ne doivent pas être isomorphes par cette application");
    // test négatif à volume ÉGAL (le premier ne diffère que par le volume) : même volume 3·4·2, couture de décalage 2 contre tore ordinaire
    let h1 = geo_libre([3, 4, 2, 1], 0, [0, 2, 0, 0]);
    let h2 = geo_libre([3, 4, 2, 1], 0, [0, 1, 0, 0]);
    assert_eq!(h1.v, h2.v);
    let r2 = std::panic::catch_unwind(|| isomorphisme(&h1, &h2));
    assert!(r2.is_err(), "sous-réseaux différents à volume égal : l'application doit être refusée");
}

// ------------------------------------------------------------------ porte 4
fn alea(etat: &mut u64) -> f64 {
    *etat = etat.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *etat;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}
fn quat(etat: &mut u64) -> Q {
    loop {
        let q = [alea(etat), alea(etat), alea(etat), alea(etat)];
        let n = qnorm(q);
        if n > 0.1 {
            return qscale(q, 1.0 / n);
        }
    }
}

#[test]
#[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
fn invariance_de_jauge_et_de_base_sur_champs_su2() {
    let (g1, g2) = deux_k(2, 3, 2);
    let phi = isomorphisme(&g1, &g2);
    let mut e = 12345u64;
    let u1: Vec<Q> = (0..4 * g1.v).map(|_| quat(&mut e)).collect();
    // image du champ : U'_μ(φ(s)) = U_μ(s)
    let mut u2 = vec![[0.0; 4]; 4 * g2.v];
    for s in 0..g1.v {
        for mu in 0..4 {
            u2[mu * g2.v + phi[s]] = u1[mu * g1.v + s];
        }
    }
    let (p1, p2) = (Pile::depuis_torsion(&g1, &Torsion::triviale()), Pile::depuis_torsion(&g2, &Torsion::triviale()));
    let (s1, s2) = (action(&g1, &u1, &p1), action(&g2, &u2, &p2));
    assert!((s1 - s2).abs() < 1e-11 * s1.abs(), "action {s1} {s2}");
    // plaquette par plaquette : les 6V plaquettes de g1 sont envoyées sur celles de g2 (mêmes traces)
    for s in 0..g1.v {
        let (x1, x2) = (g1.coords(s), g2.coords(phi[s]));
        for (mu, nu) in [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)] {
            let (a, b) = (plaquette(&g1, &u1, s, &x1, mu, nu), plaquette(&g2, &u2, phi[s], &x2, mu, nu));
            assert!(a.iter().zip(&b).all(|(p, q)| (p - q).abs() < 1e-13));
        }
    }
    // invariance de jauge sur le tore oblique : U_μ(s) → g(s) U_μ(s) g(s+μ̂)†
    let gj: Vec<Q> = (0..g1.v).map(|_| quat(&mut e)).collect();
    let mut ut = u1.clone();
    for s in 0..g1.v {
        for mu in 0..4 {
            ut[mu * g1.v + s] = qmul(qmul(gj[s], u1[mu * g1.v + s]), qdag(gj[haut(&g1, s, mu)]));
        }
    }
    let st = action(&g1, &ut, &p1);
    assert!((st - s1).abs() < 1e-11 * s1.abs(), "jauge {st} {s1}");
    // covariance de l'agrafe : A' = g(s+μ̂) A g(s)† (agrafe de tube::reseau, même route que le générateur)
    for s in 0..g1.v {
        let x = g1.coords(s);
        for mu in 0..4 {
            let a = super::reseau::agrafe(&g1, &u1, &p1, mu, s, &x);
            let at = super::reseau::agrafe(&g1, &ut, &p1, mu, s, &x);
            let att = qmul(qmul(gj[haut(&g1, s, mu)], a), qdag(gj[s]));
            assert!(at.iter().zip(&att).all(|(p, q)| (p - q).abs() < 1e-12), "agrafe non covariante s={s} mu={mu}");
        }
    }
}

// ------------------------------------------------------------------ porte 5 : énumérations exactes
/// Base HNF du sous-réseau de ℤ² engendré par g1, g2 (x, t) : (a,0), (b,d), 0 ≤ b < a, a, d > 0 (comme `controle_min_d2.py`).
fn hnf2(g1: (i64, i64), g2: (i64, i64)) -> (usize, usize, usize) {
    let (mut v1, mut v2) = ([g1.0, g1.1], [g2.0, g2.1]);
    while v2[1] != 0 {
        let q = v1[1].div_euclid(v2[1]);
        v1 = [v1[0] - q * v2[0], v1[1] - q * v2[1]];
        std::mem::swap(&mut v1, &mut v2);
    }
    let a = v2[0].abs();
    let (mut d, mut b) = (v1[1], v1[0]);
    if d < 0 {
        d = -d;
        b = -b;
    }
    (a as usize, b.rem_euclid(a) as usize, d as usize)
}

/// Tore de ℤ²×ℤ_ly (axes t, x, y, 1) de sous-réseau engendré par g1, g2 dans le plan (x, t), avec la couture décalée de kvu.
fn tore(g1: (i64, i64), g2: (i64, i64), ly: usize) -> GeoA {
    let (a, b, d) = hnf2(g1, g2);
    geo_libre([d, a, ly, 1], 0, [0, b as i64, 0, 0])
}

/// log Z de la jauge ℤ₂ (poids exp(β σ_p), σ_p = produit des 4 liens) sur les axes 0..2 de `g` (axe 3 de côté 1 ignoré) :
/// Z = 2^L cosh(β)^P Σ_{S fermée} tanh(β)^{|S|} (développement en caractères, noyau de l'incidence sur GF(2), énumération exacte).
fn lnz_jauge_z2(g: &GeoA, beta: f64) -> f64 {
    assert_eq!(g.d[3], 1);
    let v = g.v;
    assert!(3 * v <= 127);
    let mut plaq: Vec<u128> = Vec::new();
    for s in 0..v {
        for (mu, nu) in [(0, 1), (0, 2), (1, 2)] {
            let (sm, sn) = (haut(g, s, mu), haut(g, s, nu));
            let mut m = 0u128;
            for lk in [mu * v + s, nu * v + sm, mu * v + sn, nu * v + s] {
                m ^= 1u128 << lk;
            }
            plaq.push(m);
        }
    }
    let np = plaq.len();
    // noyau {S : ⊕_{p∈S} incidence(p) = 0} par élimination
    let mut piv: std::collections::HashMap<u32, (u128, u128)> = Default::default();
    let mut ker: Vec<u128> = Vec::new();
    for (i, &m0) in plaq.iter().enumerate() {
        let (mut m, mut c) = (m0, 1u128 << i);
        while m != 0 {
            let h = 127 - m.leading_zeros();
            if let Some(&(pm, pc)) = piv.get(&h) {
                m ^= pm;
                c ^= pc;
            } else {
                piv.insert(h, (m, c));
                break;
            }
        }
        if m == 0 {
            ker.push(c);
        }
    }
    assert!(ker.len() <= 26, "noyau trop gros : {}", ker.len());
    let mut w = vec![0u64; np + 1];
    let mut cur = 0u128;
    w[0] += 1;
    for i in 1u64..(1u64 << ker.len()) {
        cur ^= ker[i.trailing_zeros() as usize];
        w[cur.count_ones() as usize] += 1;
    }
    let th = beta.tanh();
    let somme: f64 = w.iter().enumerate().map(|(k, &n)| n as f64 * th.powi(k as i32)).sum();
    3.0 * v as f64 * std::f64::consts::LN_2 + np as f64 * beta.cosh().ln() + somme.ln()
}

/// log Z du modèle d'Ising (liens des axes 0 et 1, poids exp(β s s')) sur le plan de `g` (axes 2 et 3 de côté 1), par énumération.
fn lnz_ising(g: &GeoA, beta: f64) -> f64 {
    assert!(g.d[2] == 1 && g.d[3] == 1 && g.v <= 24);
    let v = g.v;
    let liens: Vec<(usize, usize)> = (0..v).flat_map(|s| [(s, haut(g, s, 0)), (s, haut(g, s, 1))]).collect();
    let mut tab = vec![0u64; liens.len() + 1]; // nombre de configurations par nombre de liens « satisfaits »
    for c in 0u32..(1u32 << v) {
        let n = liens.iter().filter(|&&(a, b)| (c >> a) & 1 == (c >> b) & 1).count();
        tab[n] += 1;
    }
    let l = liens.len() as f64;
    let z: f64 = tab.iter().enumerate().map(|(n, &k)| k as f64 * (beta * (2.0 * n as f64 - l)).exp()).sum();
    z.ln()
}

type Modele = fn(&GeoA, f64) -> f64;

fn ltore(m: Modele, g1: (i64, i64), g2: (i64, i64), ly: usize, beta: f64) -> f64 {
    m(&tore(g1, g2, ly), beta)
}

#[test]
fn valeur_de_controle_d1_seconde_difference() {
    // D-1 : D₂ = log Z(t−1) + log Z(t+1) − 2 log Z(t) = 0,00139478169078 pour R'(1,t), t = 4, L_y = 2, β = 1,2
    let (m, t, ly, beta) = (1i64, 4i64, 2usize, 1.2);
    let z = |t: i64| ltore(lnz_jauge_z2, (m, m), (t, -t), ly, beta);
    let d2 = z(t - 1) + z(t + 1) - 2.0 * z(t);
    println!("D2 = {d2:.14}");
    assert!((d2 - 0.00139478169078).abs() < 1e-11, "D2 = {d2}");
}

#[test]
fn jauge_z2_invariance_de_base_et_lemmes() {
    let beta = 0.9;
    // invariance de Z sous le changement de base unimodulaire (deux descriptions de K) : égalité à la précision machine
    for (a, b) in [(1usize, 2usize), (2, 1), (2, 2), (1, 3)] {
        let (g1, g2) = deux_k(a, b, 2);
        let (z1, z2) = (lnz_jauge_z2(&g1, beta), lnz_jauge_z2(&g2, beta));
        assert!((z1 - z2).abs() < 1e-12 * z1.abs(), "K({a},{b}) : {z1} {z2}");
        // et par engendrement direct dans le plan : ⟨(2a,0),(a,b)⟩
        let z3 = ltore(lnz_jauge_z2, (2 * a as i64, 0), (a as i64, b as i64), 2, beta);
        assert!((z1 - z3).abs() < 1e-12 * z1.abs());
    }
    // Lemme 2 : Z_K ≤ min(Z_A, Z_B)
    for (a, b) in [(1i64, 1i64), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)] {
        for beta in [0.3, 0.7, 1.2] {
            let za = ltore(lnz_jauge_z2, (2 * a, 0), (0, b), 2, beta);
            let zb = ltore(lnz_jauge_z2, (a, 0), (0, 2 * b), 2, beta);
            let zk = ltore(lnz_jauge_z2, (2 * a, 0), (a, b), 2, beta);
            assert!(zk <= za.min(zb) + 1e-12, "Z_K ≤ min(Z_A, Z_B) violé : a={a} b={b} β={beta} : {zk} {za} {zb}");
        }
    }
}

fn verifier_lemmes_3_4(m: Modele, ly: usize, mms: &[i64], ns: &[i64], betas: &[f64], ts: &[i64]) -> (usize, f64) {
    let (mut nb, mut min_d2) = (0usize, f64::INFINITY);
    for &beta in betas {
        // Lemme 3, corollaires : log-convexité des tores tournés R'(mm, t) en t
        for &mm in mms {
            for &t in ts {
                let z = |t: i64| ltore(m, (mm, mm), (t, -t), ly, beta);
                let d2 = z(t - 1) + z(t + 1) - 2.0 * z(t);
                assert!(d2 >= -1e-12, "log-convexité violée m={mm} t={t} β={beta} : {d2}");
                min_d2 = min_d2.min(d2);
                // Z(R'(m,2t)) ≤ Z(R'(m,t))²  et symétrie R'(m,t) = R'(t,m)
                assert!(z(2 * t) <= 2.0 * z(t) + 1e-12);
                let zs = ltore(m, (t, t), (mm, -mm), ly, beta);
                assert!((zs - z(t)).abs() < 1e-11 * z(t).abs().max(1.0), "symétrie R'(m,t)=R'(t,m)");
                nb += 1;
            }
        }
        // Proposition 4 : 0 ≤ a_d ≤ a_K ≤ a₃
        for &n in ns {
            let z = |g1: (i64, i64), g2: (i64, i64)| ltore(m, g1, g2, ly, beta);
            let (za, zsq) = (z((2 * n, 0), (0, n)), z((2 * n, 0), (0, 2 * n)));
            let zk = z((n, n), (n, -n));
            let zr = z((n, n), (2 * n, -2 * n));
            let (a3, ak, ad) = (2.0 * za - zsq, 2.0 * zk - zsq, 2.0 * zk - zr);
            assert!(ad >= -1e-11 && ad <= ak + 1e-11 && ak <= a3 + 1e-11, "prop 4 n={n} β={beta} : a_d={ad} a_K={ak} a3={a3}");
            nb += 1;
        }
    }
    (nb, min_d2)
}

#[test]
fn jauge_z2_log_convexite_et_proposition_4() {
    // jauge ℤ₂ 3D (L_y = 2) : tailles limitées par l'énumération exacte (noyau ≤ 26)
    let (nb, d2) = verifier_lemmes_3_4(lnz_jauge_z2, 2, &[1], &[1], &[0.3, 0.7, 1.2], &[2, 3]);
    println!("jauge Z2 3D : {nb} vérifications, min D2 = {d2:.6e}");
}

#[test]
fn ising_2d_lemmes_et_proposition_4() {
    let (nb, d2) = verifier_lemmes_3_4(lnz_ising, 1, &[1, 2], &[1, 2], &[0.2, 0.44, 0.8], &[2, 3]);
    println!("Ising 2D : {nb} vérifications, min D2 = {d2:.6e}");
}
