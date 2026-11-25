//! Côtés IMPAIRS sur carte : coloriage à trois classes (cahier O4 b, invariants I1 à I8). Branche `kvu/impairs-1120`.
//! Sans carte : I1, I2, I3, I7 (mutants de coloriage, côté hôte), I8. Avec carte (`--ignored`, sous /tmp/gamer_gpu.lock) :
//! champ froid, invariance de jauge, I4 (ordre intérieur des classes), I6 (carte = hôte), couplage fort, plaquette 9⁴/15⁴.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use kvu::commun::{BackendKind, Depart, Graine, Precision};
use kvu::geom::GeoA;
use kvu::gpu::{self, Appareil};
use kvu::heritage::su2_hb::{qdag, qmul, Q};
use kvu::su2::torsion::Torsion;
use kvu::su2::une_boite::{ParamBoite, UneBoite};
use kvu::su2::ParamChaine;

struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / 9007199254740992.0
    }
    fn gauss(&mut self) -> f64 {
        (-2.0 * self.u().ln()).sqrt() * (2.0 * std::f64::consts::PI * self.u()).cos()
    }
    fn haar(&mut self) -> Q {
        let q = [self.gauss(), self.gauss(), self.gauss(), self.gauss()];
        let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

fn qadd(a: Q, b: Q) -> Q {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

fn champ_haar(v: usize, graine: u64) -> Vec<f64> {
    let mut r = Rng(graine);
    let mut q = Vec::with_capacity(16 * v);
    for _ in 0..4 * v {
        q.extend_from_slice(&r.haar());
    }
    q
}
fn lien(v: usize, q: &[f64], mu: usize, s: usize) -> Q {
    let i = (mu * v + s) * 4;
    [q[i], q[i + 1], q[i + 2], q[i + 3]]
}
fn pose(v: usize, q: &mut [f64], mu: usize, s: usize, x: Q) {
    let i = (mu * v + s) * 4;
    q[i..i + 4].copy_from_slice(&x);
}
fn bas(g: &GeoA, s: usize, nu: usize) -> usize {
    g.decaler(s, nu, g.d[nu] - 1)
}
/// Agrafe (somme des 2·3 fers) du lien (μ, s), indexation propre au test.
fn agrafe(g: &GeoA, q: &[f64], mu: usize, s: usize) -> Q {
    let v = g.v;
    let smu = g.decaler(s, mu, 1);
    let mut a = [0.0; 4];
    for nu in 0..4 {
        if nu == mu {
            continue;
        }
        let snu = g.decaler(s, nu, 1);
        let sdn = bas(g, s, nu);
        let smd = bas(g, smu, nu);
        let t = qmul(qmul(lien(v, q, nu, smu), qdag(lien(v, q, mu, snu))), qdag(lien(v, q, nu, s)));
        let b = qmul(qmul(qdag(lien(v, q, nu, smd)), qdag(lien(v, q, mu, sdn))), lien(v, q, nu, sdn));
        a = qadd(a, qadd(t, b));
    }
    a
}
/// S_tot hôte = Σ_{μ<ν} ½ReTr U_p.
fn s_tot(g: &GeoA, q: &[f64]) -> f64 {
    let v = g.v;
    let mut tot = 0.0;
    for s in 0..v {
        for mu in 0..4 {
            for nu in mu + 1..4 {
                let a = qmul(lien(v, q, mu, s), lien(v, q, nu, g.decaler(s, mu, 1)));
                let b = qmul(qdag(lien(v, q, mu, g.decaler(s, nu, 1))), qdag(lien(v, q, nu, s)));
                tot += qmul(a, b)[0];
            }
        }
    }
    tot
}

const GEOMETRIES: [[usize; 4]; 14] = [
    [2, 2, 2, 2],
    [3, 3, 3, 3],
    [3, 3, 3, 5],
    [3, 4, 5, 7],
    [3, 5, 5, 7],
    [5, 5, 5, 5],
    [7, 4, 4, 4],
    [9, 2, 2, 2],
    [9, 9, 3, 3],
    [4, 4, 4, 4],
    [2, 3, 4, 5],
    [8, 8, 8, 8],
    [3, 6, 6, 6],
    [5, 10, 10, 10],
];

#[test]
fn i1_i2_coloriage_propre_et_couvrant() {
    // côtés 2 à 9, impairs et pairs mélangés
    for l in 2..=9usize {
        for &(a, b, c) in &[(2, 2, 2), (3, 4, 5), (l, l, l)] {
            let d = [l, a, b, c];
            let g = GeoA::nouvelle(d).unwrap();
            let cl = g.couleurs();
            g.verifier_classes(&cl).unwrap_or_else(|e| panic!("{d:?} : {e}"));
            assert_eq!(cl.len(), if g.tous_pairs() { 2 } else { 3 }, "{d:?}");
        }
    }
    for d in GEOMETRIES {
        let g = GeoA::nouvelle(d).unwrap();
        let cl = g.couleurs();
        g.verifier_classes(&cl).unwrap();
        let (mn, mx) = (cl.iter().map(Vec::len).min().unwrap(), cl.iter().map(Vec::len).max().unwrap());
        if g.tous_pairs() {
            assert_eq!(mn, mx, "damier équilibré {d:?}");
        } else {
            // équilibre approximatif (limite connue : un cycle impair a une classe plus petite) : plus grande ≤ 2 × plus petite
            assert!(mx <= 2 * mn, "{d:?} : classes {:?}", cl.iter().map(Vec::len).collect::<Vec<_>>());
        }
    }
}

/// I3 : les fers des liens (μ, s) d'une classe sont inchangés par la mise à jour de TOUS les liens (μ, ·) de cette classe.
fn independance(d: [usize; 4], classes: &[Vec<usize>]) -> bool {
    let g = GeoA::nouvelle(d).unwrap();
    let mut q = champ_haar(g.v, 99);
    let mut r = Rng(5);
    for mu in 0..4 {
        for cl in classes {
            let avant: Vec<Q> = cl.iter().map(|&s| agrafe(&g, &q, mu, s)).collect();
            for &s in cl {
                pose(g.v, &mut q, mu, s, r.haar());
            }
            for (i, &s) in cl.iter().enumerate() {
                let a = agrafe(&g, &q, mu, s);
                if (0..4).any(|k| (a[k] - avant[i][k]).abs() > 1e-13) {
                    return false;
                }
            }
        }
    }
    true
}

#[test]
fn i3_independance_des_fers() {
    for d in GEOMETRIES {
        let g = GeoA::nouvelle(d).unwrap();
        assert!(independance(d, &g.couleurs()), "{d:?}");
    }
}

#[test]
fn i7_mutants_de_coloriage_hote() {
    // (a) côté impair traité comme pair (damier) ; (b) classe 2 oubliée ; (c) deux classes fusionnées ; (d) coloriage non propre
    // au recouvrement périodique (couleur = x mod 3 sans correction du bord de cycle impair) ; (e) site dans deux classes.
    for d in [[3, 3, 3, 5], [3, 5, 5, 7], [5, 5, 5, 5], [3, 4, 4, 4]] {
        let g = GeoA::nouvelle(d).unwrap();
        let damier = || {
            let mut c = vec![Vec::new(), Vec::new()];
            for s in 0..g.v {
                let x = g.coords(s);
                c[(x[0] + x[1] + x[2] + x[3]) & 1].push(s);
            }
            c
        };
        let a = damier();
        assert!(g.verifier_classes(&a).is_err(), "(a) {d:?} : le damier doit échouer I1");
        assert!(!independance(d, &a), "(a) {d:?} : le damier doit échouer I3");
        let mut b = g.couleurs();
        b.pop();
        assert!(g.verifier_classes(&b).is_err(), "(b) {d:?}");
        let mut c = g.couleurs();
        let dernier = c.pop().unwrap();
        c[0].extend(dernier);
        c[0].sort_unstable();
        assert!(g.verifier_classes(&c).is_err(), "(c) {d:?}");
        let mut m3 = vec![Vec::new(); 3];
        for s in 0..g.v {
            let x = g.coords(s);
            m3[(x[0] + x[1] + x[2] + x[3]) % 3].push(s);
        }
        let aucun_defaut = g.verifier_classes(&m3).is_ok();
        // la somme brute mod 3 n'est propre que si L − 1 ≢ 0 (mod 3) pour chaque côté (saut du recouvrement périodique) : sinon échec
        assert_eq!(aucun_defaut, d.iter().all(|l| l % 3 != 1), "(d) {d:?}");
        let mut e = g.couleurs();
        let s0 = e[0][0];
        e[1].push(s0);
        assert!(g.verifier_classes(&e).is_err(), "(e) {d:?}");
    }
}

fn pb(cotes: [usize; 4], impairs: bool) -> ParamBoite {
    ParamBoite {
        cotes,
        torsion: Torsion::analyser("aucune").unwrap(),
        poids: 1.0,
        nor: 2,
        chaines: vec![ParamChaine { beta: 2.3, graine: Graine(1), depart: Depart::Chaud }],
        precision: Precision::F64,
        impairs,
        aniso: None,
        couture: None,
    }
}

#[test]
fn i8_refus_et_option() {
    let e = pb([3, 4, 4, 4], false).valider().unwrap_err().to_string();
    assert!(e.contains("pairs"), "{e}");
    assert!(pb([3, 4, 4, 4], true).valider().is_ok());
    assert!(pb([4, 4, 4, 4], false).valider().is_ok());
    assert!(pb([3, 4, 4, 4], true).chaines.len() == 1);
    assert!(kvu::su2::une_boite::tables_boite(&pb([3, 3, 3, 5], false)).is_err());
    assert!(kvu::su2::une_boite::tables_boite(&pb([3, 3, 3, 5], true)).is_ok());
    // pairs : le tampon geo garde le damier (geo[9] = 0)
    let (t, _, _) = kvu::su2::une_boite::tables_boite(&pb([4, 4, 4, 4], true)).unwrap();
    assert_eq!(kvu::su2::une_boite::tableau_geo_classes(&t)[9], 0);
    let (t, _, _) = kvu::su2::une_boite::tables_boite(&pb([3, 3, 3, 5], true)).unwrap();
    let g = kvu::su2::une_boite::tableau_geo_classes(&t);
    assert_eq!(g[9], 3);
    assert_eq!(g.len(), 14 + t.v);
    assert_eq!(g[13] as usize, t.v);
}

// ---------------------------------------------------------------- carte
fn appareil() -> Appareil {
    let b = if cfg!(feature = "cuda") { BackendKind::Cuda } else { BackendKind::Hip };
    gpu::ouvrir(b, 0).unwrap_or_else(|e| panic!("FAIL : backend {} indisponible ({e})", b.nom()))
}

fn pbc(cotes: [usize; 4], chaines: Vec<ParamChaine>, prec: Precision) -> ParamBoite {
    ParamBoite {
        cotes,
        torsion: Torsion::analyser("aucune").unwrap(),
        poids: 1.0,
        nor: 2,
        chaines,
        precision: prec,
        impairs: true,
        aniso: None,
        couture: None,
    }
}
fn ch(beta: f64, g: u64, d: Depart) -> ParamChaine {
    ParamChaine { beta, graine: Graine(g), depart: d }
}
fn mesure_carte(m: &mut UneBoite, c: usize) -> f64 {
    m.allouer_historique(1).unwrap();
    m.mesurer(0).unwrap();
    m.historique().unwrap()[2 * c]
}

#[test]
#[ignore = "GPU"]
fn g_champ_froid_jauge_et_hote() {
    let app = appareil();
    for d in [[3, 3, 3, 3], [3, 3, 3, 5], [3, 5, 5, 7], [5, 5, 5, 5], [7, 4, 4, 4], [9, 2, 2, 2]] {
        let g = GeoA::nouvelle(d).unwrap();
        // champ froid : S_tot = 6 V exactement (aucun balayage)
        let mut m = UneBoite::nouveau(&app, &pbc(d, vec![ch(2.3, 1, Depart::Froid)], Precision::F64)).unwrap();
        m.init().unwrap();
        let s = mesure_carte(&mut m, 0);
        assert!((s - 6.0 * g.v as f64).abs() < 1e-9 * g.v as f64, "froid {d:?} : {s}");
        // thermalisation puis : carte = hôte (I6) et invariance de jauge
        let mut m = UneBoite::nouveau(&app, &pbc(d, vec![ch(2.3, 3, Depart::Chaud), ch(2.3, 4, Depart::Froid)], Precision::F64)).unwrap();
        m.init().unwrap();
        for _ in 0..12 {
            m.pas().unwrap();
        }
        for c in 0..2 {
            let q = m.telecharger(c).unwrap();
            for s in 0..g.v {
                for mu in 0..4 {
                    let l = lien(g.v, &q, mu, s);
                    let n: f64 = l.iter().map(|x| x * x).sum();
                    assert!((n - 1.0).abs() < 1e-12, "non unitaire {d:?}");
                }
            }
            let hote = s_tot(&g, &q);
            let carte = mesure_carte(&mut m, c);
            assert!((carte - hote).abs() < 1e-9 * hote.abs(), "I6 {d:?} : carte {carte} hôte {hote}");
            // transformation de jauge aléatoire : U_μ(x) → g(x) U_μ(x) g(x+μ)†
            let mut r = Rng(77 + c as u64);
            let gs: Vec<Q> = (0..g.v).map(|_| r.haar()).collect();
            let mut q2 = q.clone();
            for s in 0..g.v {
                for mu in 0..4 {
                    let n = qmul(qmul(gs[s], lien(g.v, &q, mu, s)), qdag(gs[g.decaler(s, mu, 1)]));
                    pose(g.v, &mut q2, mu, s, n);
                }
            }
            let hote2 = s_tot(&g, &q2);
            assert!((hote2 - hote).abs() < 1e-9 * hote.abs(), "jauge hôte {d:?}");
            m.televerser(c, &q2).unwrap();
            let carte2 = mesure_carte(&mut m, c);
            assert!((carte2 - carte).abs() < 1e-10 * carte.abs(), "jauge carte {d:?} : {carte2} contre {carte}");
        }
        println!("PASS froid/jauge/hôte {d:?}");
    }
}

#[test]
#[ignore = "GPU"]
fn g_i4_ordre_interieur_des_classes() {
    // Le résultat d'un balayage ne dépend pas de l'ordre des sites dans une classe (clé Philox par lien, fers indépendants) :
    // classes mélangées (graine fixe) ⇒ champ IDENTIQUE AU BIT. Aucun mutant de classes ne doit lui ressembler.
    let app = appareil();
    for (d, prec) in [([3, 3, 3, 5], Precision::F64), ([3, 5, 5, 7], Precision::F64), ([5, 5, 5, 5], Precision::F32), ([3, 4, 4, 4], Precision::F64)] {
        let g = GeoA::nouvelle(d).unwrap();
        let p = pbc(d, vec![ch(2.0, 5, Depart::Chaud), ch(2.6, 6, Depart::Chaud)], prec);
        let mut a = UneBoite::nouveau(&app, &p).unwrap();
        let mut cl = g.couleurs();
        let mut r = Rng(31);
        for c in cl.iter_mut() {
            for i in (1..c.len()).rev() {
                let j = (r.u() * (i + 1) as f64) as usize % (i + 1);
                c.swap(i, j);
            }
        }
        let mut b = UneBoite::nouveau_classes(&app, &p, cl).unwrap();
        a.init().unwrap();
        b.init().unwrap();
        for _ in 0..6 {
            a.pas().unwrap();
            b.pas().unwrap();
        }
        for c in 0..2 {
            assert_eq!(a.telecharger(c).unwrap(), b.telecharger(c).unwrap(), "I4 {d:?}");
        }
        // mutant : un côté impair traité comme pair (damier) donne un autre champ ET n'est pas un coloriage propre
        if !g.tous_pairs() {
            let mut dam = vec![Vec::new(), Vec::new()];
            for s in 0..g.v {
                let x = g.coords(s);
                dam[(x[0] + x[1] + x[2] + x[3]) & 1].push(s);
            }
            assert!(g.verifier_classes(&dam).is_err());
            let mut mu = UneBoite::nouveau_classes(&app, &p, dam).unwrap();
            mu.init().unwrap();
            for _ in 0..6 {
                mu.pas().unwrap();
            }
            assert_ne!(a.telecharger(0).unwrap(), mu.telecharger(0).unwrap(), "le mutant damier doit différer");
        }
        println!("PASS I4 {d:?}");
    }
}

/// ⟨½Tr U_p⟩ moyenne et erreur (réplicas).
fn plaquette(app: &Appareil, d: [usize; 4], beta: f64, r: usize, therm: usize, nmes: usize, graine: u64, prec: Precision) -> (f64, f64) {
    let v = d.iter().product::<usize>() as f64;
    let chs: Vec<ParamChaine> = (0..r).map(|i| ch(beta, graine + i as u64, if i % 2 == 0 { Depart::Froid } else { Depart::Chaud })).collect();
    let mut m = UneBoite::nouveau(app, &pbc(d, chs, prec)).unwrap();
    m.init().unwrap();
    let s = m.serie(therm, nmes, 1).unwrap();
    assert!(m.compteurs().unwrap().iter().all(|c| c.0 == 0), "plafond du bain atteint");
    let moy: Vec<f64> = (0..r).map(|c| s[c].iter().map(|x| x[0]).sum::<f64>() / nmes as f64 / (6.0 * v)).collect();
    let mu = moy.iter().sum::<f64>() / r as f64;
    let va = moy.iter().map(|y| (y - mu).powi(2)).sum::<f64>() / (r - 1) as f64;
    (mu, (va / r as f64).sqrt())
}

/// I2/I1 de Bessel (série), moyenne ⟨½Tr U⟩ pour le poids exp(β ½ Tr U) sur SU(2) = I2(β)/I1(β) (premier ordre du couplage fort).
fn i2_sur_i1(b: f64) -> f64 {
    let i = |nu: i32| -> f64 {
        let mut t = (b / 2.0).powi(nu) / (1..=nu).map(|k| k as f64).product::<f64>();
        let mut s = t;
        for k in 1..60 {
            t *= (b / 2.0).powi(2) / (k as f64 * (k + nu) as f64);
            s += t;
        }
        s
    };
    i(2) / i(1)
}

#[test]
#[ignore = "GPU"]
fn g_couplage_fort_et_beta_zero() {
    let app = appareil();
    // β = 0 : Haar, ⟨½Tr U_p⟩ = 0
    for d in [[3, 3, 3, 5], [5, 5, 5, 5]] {
        let (m, e) = plaquette(&app, d, 0.0, 16, 30, 200, 11, Precision::F64);
        println!("β=0 {d:?} : {m:.5} ± {e:.5}");
        assert!(m.abs() < 5.0 * e + 1e-12, "β=0 {d:?}");
    }
    // couplage fort : P = I2/I1(β) + O(β⁵) (graine unique ; correction de cube non nulle en 4D à l'ordre β⁵)
    for d in [[3, 3, 3, 3], [5, 5, 5, 5], [3, 3, 3, 5]] {
        for beta in [0.2, 0.4] {
            let (m, e) = plaquette(&app, d, beta, 16, 60, 400, 21, Precision::F64);
            let th = i2_sur_i1(beta);
            println!("fort {d:?} β={beta} : {m:.6} ± {e:.6}  I2/I1 = {th:.6}  (β/4 = {:.6})", beta / 4.0);
            assert!((m - th).abs() < 5.0 * e + 12.0 * beta.powi(5), "{d:?} β={beta}");
        }
    }
}

#[test]
#[ignore = "GPU"]
fn g_plaquette_impairs_contre_pairs() {
    let app = appareil();
    let g = |d: [usize; 4], r, th, nm, s| plaquette(&app, d, 2.3, r, th, nm, s, Precision::F64);
    let (p8, e8) = g([8; 4], 32, 1500, 2500, 7_100_000);
    let (p9, e9) = g([9; 4], 32, 1500, 2500, 7_200_000);
    let (p10, e10) = g([10; 4], 32, 1500, 2500, 7_300_000);
    let (p15, e15) = g([15; 4], 16, 1000, 1500, 7_400_000);
    let (p16, e16) = g([16; 4], 16, 1000, 1500, 7_500_000);
    println!("β=2,3 : 8⁴ {p8:.7}({:.0}e-7)  9⁴ {p9:.7}({:.0}e-7)  10⁴ {p10:.7}({:.0}e-7)  15⁴ {p15:.7}({:.0}e-7)  16⁴ {p16:.7}({:.0}e-7)", e8 * 1e7, e9 * 1e7, e10 * 1e7, e15 * 1e7, e16 * 1e7);
    // l'effet de taille finie est exponentiellement petit à β = 2,3 sur ces tailles : 9 compatible avec 8 et 10, 15 avec 16
    let z = |a: f64, ea: f64, b: f64, eb: f64| (a - b) / (ea * ea + eb * eb).sqrt();
    println!("z(9-8) = {:+.2}, z(9-10) = {:+.2}, z(15-16) = {:+.2}", z(p9, e9, p8, e8), z(p9, e9, p10, e10), z(p15, e15, p16, e16));
    assert!(z(p9, e9, p8, e8).abs() < 4.0 && z(p9, e9, p10, e10).abs() < 4.0 && z(p15, e15, p16, e16).abs() < 4.0);
}
