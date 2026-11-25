//! Tests de `su2::une_boite` (campagne « une boîte »). Critères : `deux_voies_2026-11-16/kvu_campagne/PREDICTIONS_KVU.md` (haché avant).
//! Sans carte : C1 à C3. Avec carte (`--ignored`, sous /tmp/gamer_gpu.lock, tranches ≤ 5 min) : G1 à G9.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use kvu::commun::{BackendKind, Depart, Graine, Precision};
use kvu::geom::{GeoA, Pile};
use kvu::gpu::{self, Appareil, OptionsCompil};
use kvu::heritage::su2_hb::{qdag, qmul, Q};
use kvu::su2::geo_gpu::{spec_su2_gen, Geo64Gpu};
use kvu::su2::torsion::Torsion;
use kvu::su2::une_boite::{tables_boite, ParamBoite, UneBoite};
use kvu::su2::{ConfigLot, NoyauxSu2, ParamChaine};

// ---------------------------------------------------------------- outils de test (indexation PROPRE au test)
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

fn nsites(d: [usize; 4]) -> usize {
    d.iter().product()
}
fn idx(d: [usize; 4], x: [usize; 4]) -> usize {
    ((x[0] * d[1] + x[1]) * d[2] + x[2]) * d[3] + x[3]
}
fn coords(d: [usize; 4], s: usize) -> [usize; 4] {
    let mut x = [0; 4];
    let mut t = s;
    for k in (0..4).rev() {
        x[k] = t % d[k];
        t /= d[k];
    }
    x
}
fn haut(d: [usize; 4], x: [usize; 4], mu: usize) -> usize {
    let mut y = x;
    y[mu] = (y[mu] + 1) % d[mu];
    idx(d, y)
}
fn champ_haar(d: [usize; 4], graine: u64) -> Vec<f64> {
    let mut r = Rng(graine);
    let v = nsites(d);
    let mut q = Vec::with_capacity(16 * v);
    for _ in 0..4 * v {
        q.extend_from_slice(&r.haar());
    }
    q
}
fn lien(d: [usize; 4], q: &[f64], mu: usize, s: usize) -> Q {
    let i = (mu * nsites(d) + s) * 4;
    [q[i], q[i + 1], q[i + 2], q[i + 3]]
}
const PLANS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// (S_tot, S_feuille) avec une feuille d'ORIGINE `o` (coin (o_μ, o_ν)) dans les plans `plans` (indices de PLANS).
fn mesure_hote(d: [usize; 4], q: &[f64], plans: &[usize], o: [usize; 4]) -> (f64, f64) {
    let (mut tot, mut fe) = (0.0, 0.0);
    for s in 0..nsites(d) {
        let x = coords(d, s);
        for (p, &(mu, nu)) in PLANS.iter().enumerate() {
            let a = qmul(lien(d, q, mu, s), lien(d, q, nu, haut(d, x, mu)));
            let b = qmul(
                qdag(lien(d, q, mu, haut(d, x, nu))),
                qdag(lien(d, q, nu, s)),
            );
            let v = qmul(a, b)[0];
            tot += v;
            if plans.contains(&p) && x[mu] == o[mu] && x[nu] == o[nu] {
                fe += v;
            }
        }
    }
    (tot, fe)
}

fn pb(cotes: [usize; 4], tors: &str, poids: f64, chaines: Vec<ParamChaine>) -> ParamBoite {
    pbp(cotes, tors, poids, chaines, Precision::F64)
}
fn pbp(
    cotes: [usize; 4],
    tors: &str,
    poids: f64,
    chaines: Vec<ParamChaine>,
    precision: Precision,
) -> ParamBoite {
    ParamBoite {
        cotes,
        torsion: Torsion::analyser(tors).unwrap(),
        poids,
        nor: 2,
        chaines,
        precision,
        impairs: false,
        aniso: None,
        couture: None,
    }
}
fn ch(beta: f64, g: u64, d: Depart) -> ParamChaine {
    ParamChaine {
        beta,
        graine: Graine(g),
        depart: d,
    }
}

// ---------------------------------------------------------------- sans carte
/// Vérifie qu'un masque de feuille est EXACTEMENT la pile attendue.
fn pile_conforme(cotes: [usize; 4], plans: &[usize], feuille: &[u8]) -> bool {
    let v = nsites(cotes);
    for p in 0..6 {
        let (mu, nu) = PLANS[p];
        for s in 0..v {
            let x = coords(cotes, s);
            let attendu = plans.contains(&p) && x[mu] == cotes[mu] - 1 && x[nu] == cotes[nu] - 1;
            if (feuille[p * v + s] != 0) != attendu {
                return false;
            }
        }
    }
    true
}

#[test]
fn c1_pile_rectangulaire_et_mutants() {
    for (cotes, tors, plans) in [
        ([4, 6, 8, 2], "01", vec![0usize]),
        ([4, 6, 8, 2], "23", vec![5usize]),
        ([2, 4, 2, 4], "02,13", vec![1, 4]),
    ] {
        let (t, coef, feuille) =
            tables_boite(&pb(cotes, tors, -1.0, vec![ch(2.0, 1, Depart::Froid)])).unwrap();
        assert!(
            pile_conforme(cotes, &plans, &feuille),
            "pile fausse {cotes:?} {tors}"
        );
        for &p in &plans {
            let n: usize = feuille[p * t.v..(p + 1) * t.v]
                .iter()
                .map(|&b| b as usize)
                .sum();
            let (mu, nu) = PLANS[p];
            assert_eq!(
                n,
                t.v / (cotes[mu] * cotes[nu]),
                "nombre de plaquettes de feuille"
            );
        }
        let c = coef.unwrap();
        assert!(c
            .iter()
            .zip(&feuille)
            .all(|(&z, &f)| z == if f != 0 { -1.0 } else { 1.0 }));
        // poids réel
        let (_, coef2, f2) =
            tables_boite(&pb(cotes, tors, 0.37, vec![ch(2.0, 1, Depart::Froid)])).unwrap();
        assert!(coef2
            .unwrap()
            .iter()
            .zip(&f2)
            .all(|(&z, &f)| z == if f != 0 { 0.37 } else { 1.0 }));
        // MUTANTS : le vérificateur doit refuser
        let mut decale = feuille.clone();
        let p0 = plans[0];
        let v = t.v;
        let i = (0..v).find(|&s| decale[p0 * v + s] != 0).unwrap();
        decale[p0 * v + i] = 0;
        decale[p0 * v + (i + 1) % v] = 1; // une plaquette déplacée d'un site
        assert!(
            !pile_conforme(cotes, &plans, &decale),
            "mutant « pile décalée » non détecté"
        );
        let mut incomplet = feuille.clone();
        incomplet[p0 * v + i] = 0;
        assert!(
            !pile_conforme(cotes, &plans, &incomplet),
            "mutant « feuille incomplète » non détecté"
        );
        let autre: Vec<usize> = plans.iter().map(|&p| (p + 1) % 6).collect();
        assert!(
            !pile_conforme(cotes, &autre, &feuille),
            "mutant « mauvais plan » non détecté"
        );
    }
    // la pile des tables est celle de geom::Pile (pile de kvu) : même ensemble
    let g = GeoA::nouvelle([4, 6, 8, 2]).unwrap();
    let pile = Pile::depuis_torsion(&g, &Torsion::analyser("01").unwrap());
    let (_, _, f) = tables_boite(&pb(
        [4, 6, 8, 2],
        "01",
        -1.0,
        vec![ch(2.0, 1, Depart::Froid)],
    ))
    .unwrap();
    assert!((0..g.v).all(|s| (pile.z(0, s) < 0.0) == (f[s] != 0)));
}

#[test]
fn c2_pile_deplacee_identite_exacte_et_mutant() {
    for (d, plan) in [
        ([4usize, 4, 6, 2], 0usize),
        ([4, 4, 6, 2], 5),
        ([2, 4, 2, 4], 3),
        ([4, 6, 2, 4], 2),
    ] {
        let (mu, nu) = PLANS[plan];
        let q0 = champ_haar(d, 99 + plan as u64);
        let ancien = [d[0] - 1, d[1] - 1, d[2] - 1, d[3] - 1]; // convention kvu : coin (ℓ−1, ℓ−1)
        let (tot0, fe0) = mesure_hote(d, &q0, &[plan], ancien);
        let qs0 = (tot0 - fe0) - fe0; // Q_{-1} = S_R − S_F
                                      // transformation : pile (a, b) → (a+1, b+1) mod ℓ (a = ℓ_μ−1, b = ℓ_ν−1)
        let (a, b) = (ancien[mu], ancien[nu]);
        let (a1, b1) = ((a + 1) % d[mu], (b + 1) % d[nu]);
        let mut q1 = q0.clone();
        let v = nsites(d);
        let flip = |q: &mut Vec<f64>, dir: usize, cond: &dyn Fn([usize; 4]) -> bool| {
            for s in 0..v {
                if cond(coords(d, s)) {
                    let i = (dir * v + s) * 4;
                    for k in 0..4 {
                        q[i + k] = -q[i + k];
                    }
                }
            }
        };
        // 1) déplacement en μ : liens U_ν à y_μ = a+1, y_ν = b
        flip(&mut q1, nu, &|y| y[mu] == a1 && y[nu] == b);
        // 2) déplacement en ν : liens U_μ à y_ν = b+1, y_μ = a+1 (la pile est maintenant en (a+1, b))
        flip(&mut q1, mu, &|y| y[nu] == b1 && y[mu] == a1);
        let mut o = ancien;
        o[mu] = a1;
        o[nu] = b1;
        let (tot1, fe1) = mesure_hote(d, &q1, &[plan], o);
        let qs1 = (tot1 - fe1) - fe1;
        assert!(
            (qs0 - qs1).abs() < 1e-12 * (1.0 + qs0.abs()),
            "pile déplacée : Q {qs0} contre {qs1} (plan {plan}, côtés {d:?})"
        );
        // MUTANT : sans la transformation, la feuille déplacée donne une autre valeur
        let (tm, fm) = mesure_hote(d, &q0, &[plan], o);
        let qm = (tm - fm) - fm;
        assert!(
            (qs0 - qm).abs() > 1e-3,
            "mutant « pas de transformation » non détecté ({qs0} {qm})"
        );
    }
}

#[test]
fn c3_refus() {
    let un = || vec![ch(2.0, 1, Depart::Froid)];
    assert!(
        pb([3, 4, 4, 4], "01", -1.0, un()).valider().is_err(),
        "côté impair"
    );
    assert!(
        pb([4, 0, 4, 4], "01", -1.0, un()).valider().is_err(),
        "côté 0"
    );
    assert!(
        pb([4, 4, 4, 4], "01", -1.0, vec![ch(-1.0, 1, Depart::Froid)])
            .valider()
            .is_err(),
        "β < 0"
    );
    assert!(
        pb([4, 4, 4, 4], "01", f64::NAN, un()).valider().is_err(),
        "poids NaN"
    );
    assert!(
        pb([4, 4, 4, 4], "01", -1.0, vec![]).valider().is_err(),
        "0 chaîne"
    );
    assert!(pb([4, 4, 4, 4], "01", -1.0, un()).valider().is_ok());
}

// ---------------------------------------------------------------- avec carte
fn appareil() -> Appareil {
    let b = if cfg!(feature = "cuda") {
        BackendKind::Cuda
    } else {
        BackendKind::Hip
    };
    gpu::ouvrir(b, 0).unwrap_or_else(|e| panic!("FAIL : backend {} indisponible ({e})", b.nom()))
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / (1.0 + b.abs())
}

#[test]
#[ignore = "GPU"]
fn g1_mesure_egale_hote() {
    let app = appareil();
    for (cotes, tors, plans) in [
        ([4, 4, 4, 4], "01", vec![0usize]),
        ([2, 4, 6, 8], "02,13", vec![1, 4]),
        ([4, 8, 8, 8], "01", vec![0]),
    ] {
        for poids in [-1.0, 1.0] {
            let mut m = UneBoite::nouveau(
                &app,
                &pb(
                    cotes,
                    tors,
                    poids,
                    vec![ch(2.3, 5, Depart::Froid), ch(2.3, 6, Depart::Chaud)],
                ),
            )
            .unwrap();
            m.init().unwrap();
            let q = [champ_haar(cotes, 1234), champ_haar(cotes, 4321)];
            for c in 0..2 {
                m.televerser(c, &q[c]).unwrap();
            }
            m.allouer_historique(1).unwrap();
            m.mesurer(0).unwrap();
            let h = m.historique().unwrap();
            let o = [cotes[0] - 1, cotes[1] - 1, cotes[2] - 1, cotes[3] - 1];
            for c in 0..2 {
                // chaque plan tordu a son propre coin : on somme plan par plan
                let (mut tot, mut fe) = (0.0, 0.0);
                tot += mesure_hote(cotes, &q[c], &[], o).0;
                for &p in &plans {
                    fe += mesure_hote(cotes, &q[c], &[p], o).1;
                }
                assert!(
                    rel(h[2 * c], tot) < 1e-11,
                    "S_tot {} contre {tot} ({cotes:?})",
                    h[2 * c]
                );
                assert!(
                    rel(h[2 * c + 1], fe) < 1e-11,
                    "S_F {} contre {fe} ({cotes:?})",
                    h[2 * c + 1]
                );
            }
        }
    }
    println!("PASS G1");
}

#[test]
#[ignore = "GPU"]
fn g2_identite_au_bit_avec_geo64() {
    let app = appareil();
    for tors in ["aucune", "01"] {
        let chs = vec![
            ch(2.3, 11, Depart::Froid),
            ch(2.3, 12, Depart::Chaud),
            ch(2.3, 13, Depart::Chaud),
        ];
        let mut m = UneBoite::nouveau(&app, &pb([4, 4, 4, 4], tors, -1.0, chs.clone())).unwrap();
        let cfg = ConfigLot {
            l: 4,
            chaines: chs,
            nor: 2,
            precision: Precision::F64,
            options: OptionsCompil::reference_k1(),
            mutant_metro: false,
        };
        let spec = spec_su2_gen(4, Torsion::analyser(tors).unwrap());
        let mut g = Geo64Gpu::nouveau(&app, cfg, &spec).unwrap();
        m.init().unwrap();
        g.init().unwrap();
        for _ in 0..12 {
            m.pas().unwrap();
            g.sweep_hb().unwrap();
            g.sweep_or().unwrap();
            g.sweep_or().unwrap();
        }
        for c in 0..3 {
            let a = m.telecharger(c).unwrap();
            let b = g.telecharger(c).unwrap().q;
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "torsion {tors}, chaîne {c} : champs différents au bit"
            );
        }
        println!("PASS G2 torsion {tors}");
    }
}

/// moyenne et erreur (entre répliques) de P = S_tot/(6V) ; moitiés
fn plaquette_repliques(
    app: &Appareil,
    cotes: [usize; 4],
    beta: f64,
    r: usize,
    therm: usize,
    nmes: usize,
    graine: u64,
) -> (f64, f64, f64, f64, f64) {
    plaquette_repliques_p(app, cotes, beta, r, therm, nmes, graine, Precision::F64)
}

#[allow(clippy::too_many_arguments)]
fn plaquette_repliques_p(
    app: &Appareil,
    cotes: [usize; 4],
    beta: f64,
    r: usize,
    therm: usize,
    nmes: usize,
    graine: u64,
    prec: Precision,
) -> (f64, f64, f64, f64, f64) {
    let chs: Vec<ParamChaine> = (0..r)
        .map(|i| {
            ch(
                beta,
                graine + i as u64,
                if i % 2 == 0 {
                    Depart::Froid
                } else {
                    Depart::Chaud
                },
            )
        })
        .collect();
    let t0 = std::time::Instant::now();
    let mut m = UneBoite::nouveau(app, &pbp(cotes, "aucune", 1.0, chs, prec)).unwrap();
    m.init().unwrap();
    let s = m.serie(therm, nmes, 1).unwrap();
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "débit {:?} {:?} : {:.3e} liens/s ({} chaînes, {:.1} s)",
        prec,
        cotes,
        (therm + nmes) as f64 * 3.0 * 4.0 * nsites(cotes) as f64 * r as f64 / dt,
        r,
        dt
    );
    assert!(
        m.compteurs().unwrap().iter().all(|c| c.0 == 0),
        "plafond du bain atteint"
    );
    let v6 = 6.0 * nsites(cotes) as f64;
    let moy = |c: usize, a: usize, b: usize| {
        s[c][a..b].iter().map(|x| x[0]).sum::<f64>() / (b - a) as f64 / v6
    };
    let tous: Vec<f64> = (0..r).map(|c| moy(c, 0, nmes)).collect();
    let h1: Vec<f64> = (0..r).map(|c| moy(c, 0, nmes / 2)).collect();
    let h2: Vec<f64> = (0..r).map(|c| moy(c, nmes / 2, nmes)).collect();
    let stat = |x: &[f64]| {
        let m = x.iter().sum::<f64>() / x.len() as f64;
        let v = x.iter().map(|y| (y - m).powi(2)).sum::<f64>() / (x.len() - 1) as f64;
        (m, (v / x.len() as f64).sqrt())
    };
    let (m, e) = stat(&tous);
    let (a, ea) = stat(&h1);
    let (b, eb) = stat(&h2);
    (m, e, a - b, (ea * ea + eb * eb).sqrt(), 0.0)
}

fn porte_plaquette(
    cotes: [usize; 4],
    beta: f64,
    r: usize,
    therm: usize,
    nmes: usize,
    refm: f64,
    refe: f64,
) {
    let app = appareil();
    let (m, e, dh, edh, _) = plaquette_repliques(
        &app,
        cotes,
        beta,
        r,
        therm,
        nmes,
        7_000_000 + (beta * 100.0) as u64,
    );
    let z = (m - refm) / (e * e + refe * refe).sqrt();
    println!("G3 {cotes:?} β={beta} : P = {m:.7}({:.0}e-7) réf {refm:.7} z = {z:+.2} ; moitiés Δ = {dh:+.2e} ± {edh:.1e}", e * 1e7);
    assert!(z.abs() <= 4.0, "z = {z}");
    assert!(
        dh.abs() <= 4.0 * edh,
        "moitiés incompatibles : {dh} ± {edh}"
    );
}

#[test]
#[ignore = "GPU"]
fn g3_plaquette_8_230() {
    porte_plaquette([8; 4], 2.30, 32, 1500, 2500, 0.6022749, 1.6e-6);
}
#[test]
#[ignore = "GPU"]
fn g3_plaquette_8_240() {
    porte_plaquette([8; 4], 2.40, 32, 1500, 2500, 0.6302997, 1.4e-6);
}
#[test]
#[ignore = "GPU"]
fn g3_plaquette_16_230() {
    porte_plaquette([16; 4], 2.30, 16, 1500, 2000, 0.6022641, 1.5e-6);
}
#[test]
#[ignore = "GPU"]
fn g3_plaquette_16_240() {
    porte_plaquette([16; 4], 2.40, 16, 1500, 2000, 0.6299985, 1.2e-6);
}

#[test]
#[ignore = "GPU"]
fn g4_rectangulaire_contre_tube_cpu() {
    use kvu::su2::tube::reseau::ChaineTube;
    let app = appareil();
    for (cotes, tors) in [
        ([4, 4, 4, 8], "01"),
        ([4, 4, 4, 8], "aucune"),
        ([2, 4, 6, 4], "02"),
    ] {
        let t = Torsion::analyser(tors).unwrap();
        let mut m = UneBoite::nouveau(
            &app,
            &pb(cotes, tors, -1.0, vec![ch(2.3, 777, Depart::Chaud)]),
        )
        .unwrap();
        m.init().unwrap();
        let g = GeoA::nouvelle(cotes).unwrap();
        let pile = Pile::depuis_torsion(&g, &t);
        let mut c = ChaineTube::nouvelle(g, 777);
        c.init(true);
        let a0 = m.telecharger(0).unwrap();
        let ecart_init = a0
            .iter()
            .zip(c.u.iter().flatten())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(ecart_init < 1e-12, "départ chaud différent : {ecart_init}");
        m.sweep_hb().unwrap();
        c.sweep_hb(2.3, &pile);
        let a = m.telecharger(0).unwrap();
        let e = a
            .iter()
            .zip(c.u.iter().flatten())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        println!("G4 {cotes:?} {tors} : un bain, max|Δq| = {e:.2e}");
        assert!(e <= 1e-11, "un balayage de bain : écart {e}");
    }
}

/// (ln R, σ) aux β cibles par la méthode directe (nœuds équidistants, Simpson et trapèze), répliques appariées +/−.
#[allow(clippy::type_complexity)]
fn direct_l2(
    app: &Appareil,
    cotes: [usize; 4],
    tors: &str,
    h: f64,
    nnoeuds: usize,
    r: usize,
    therm: usize,
    nmes: usize,
    poids_tordu: f64,
    mutant: u8,
) -> (Vec<f64>, Vec<(f64, f64)>, Vec<f64>) {
    let betas: Vec<f64> = (1..=nnoeuds).map(|k| h * k as f64).collect();
    let chs = |off: u64| -> Vec<ParamChaine> {
        betas
            .iter()
            .enumerate()
            .flat_map(|(k, &b)| {
                (0..r).map(move |i| {
                    ch(
                        b,
                        off + (k * r + i) as u64,
                        if i % 2 == 0 {
                            Depart::Froid
                        } else {
                            Depart::Chaud
                        },
                    )
                })
            })
            .collect()
    };
    // graines APPARIÉES : mêmes graines pour le bras tordu et le témoin
    let mut g_par_noeud = vec![(0.0, 0.0); nnoeuds];
    let mut bras = vec![];
    for poids in [poids_tordu, 1.0] {
        let mut m =
            UneBoite::nouveau_mutant(app, &pb(cotes, tors, poids, chs(31_000)), mutant).unwrap();
        m.init().unwrap();
        bras.push(m.serie(therm, nmes, 1).unwrap());
    }
    // g_r = ⟨Q_s⟩_tordu − ⟨Q_+⟩_témoin par réplique ; Q_s = S_R + s S_F = S_tot + (s−1) S_F
    let mut g_rep = vec![vec![0.0; r]; nnoeuds];
    for k in 0..nnoeuds {
        for i in 0..r {
            let c = k * r + i;
            let q = |b: usize, s: f64| {
                bras[b][c]
                    .iter()
                    .map(|x| x[0] + (s - 1.0) * x[1])
                    .sum::<f64>()
                    / nmes as f64
            };
            g_rep[k][i] = q(0, poids_tordu) - q(1, 1.0);
        }
    }
    for k in 0..nnoeuds {
        let m = g_rep[k].iter().sum::<f64>() / r as f64;
        let v = g_rep[k].iter().map(|y| (y - m).powi(2)).sum::<f64>() / (r - 1) as f64;
        g_par_noeud[k] = (m, (v / r as f64).sqrt());
    }
    // intégrales cumulées : nœud 0 (β=0) g=0 analytique
    let mut simpson = vec![f64::NAN; nnoeuds];
    let mut trapeze = vec![f64::NAN; nnoeuds];
    let mut var_s = vec![0.0; nnoeuds];
    let gk = |k: isize| {
        if k < 0 {
            0.0
        } else {
            g_par_noeud[k as usize].0
        }
    };
    for k in 0..nnoeuds {
        // trapèze depuis β = 0
        let mut t = 0.5 * h * (0.0 + gk(0));
        for j in 1..=k {
            t += 0.5 * h * (gk(j as isize - 1) + gk(j as isize));
        }
        trapeze[k] = t;
        // Simpson sur les intervalles pairs (nœuds -1 = β 0), le dernier intervalle impair par trapèze corrigé
        let n = k + 1; // nombre d'intervalles de β=0 à β_k
        let mut s = 0.0;
        let mut w = vec![0.0; nnoeuds];
        let mut j = 0usize;
        while j + 2 <= n {
            // intervalle [j, j+2] en indices de nœuds décalés (0 ↔ β=0)
            let (a, b, c) = (j as isize - 1, j as isize, j as isize + 1);
            for (idx, coef) in [(a, h / 3.0), (b, 4.0 * h / 3.0), (c, h / 3.0)] {
                if idx >= 0 {
                    s += coef * g_par_noeud[idx as usize].0;
                    w[idx as usize] += coef;
                }
            }
            j += 2;
        }
        if j < n {
            // un intervalle restant : parabole sur les trois derniers points (β_{k-2}, β_{k-1}, β_k) intégrée sur le dernier pas : poids 5/12, 8/12, −1/12 (h)
            let (a, b, c) = (k as isize - 2, k as isize - 1, k as isize);
            for (idx, coef) in [(a, -h / 12.0), (b, 8.0 * h / 12.0), (c, 5.0 * h / 12.0)] {
                if idx >= 0 {
                    s += coef * g_par_noeud[idx as usize].0;
                    w[idx as usize] += coef;
                }
            }
            if k == 0 {
                s = 0.5 * h * gk(0);
                w = vec![0.0; nnoeuds];
                w[0] = 0.5 * h;
            }
        }
        simpson[k] = s;
        var_s[k] = (0..nnoeuds)
            .map(|i| (w[i] * g_par_noeud[i].1).powi(2))
            .sum::<f64>();
    }
    (
        betas,
        simpson
            .iter()
            .zip(&var_s)
            .map(|(&s, &v)| (s, v.sqrt()))
            .collect(),
        trapeze,
    )
}

const REFS_L2: [(f64, f64, f64); 3] = [
    (1.0, -0.03285, 0.01078),
    (2.0, -1.25154, 0.01998),
    (2.5, -2.54268, 0.02265),
];

#[test]
#[ignore = "GPU"]
fn g5_methode_directe_l2() {
    let app = appareil();
    let (b, s, t) = direct_l2(&app, [2; 4], "23", 0.125, 20, 256, 400, 4000, -1.0, 0);
    for (beta, refv, refe) in REFS_L2 {
        let k = b.iter().position(|x| (x - beta).abs() < 1e-9).unwrap();
        let (v, e) = s[k];
        let z = (v - refv) / (e * e + refe * refe).sqrt();
        println!(
            "G5 β={beta} : Simpson {v:+.5}({e:.5}) trapèze {:+.5} réf {refv}({refe}) z = {z:+.2}",
            t[k]
        );
        assert!(z.abs() <= 3.0, "z = {z}");
        assert!(
            (v - t[k]).abs() <= 0.01,
            "biais de quadrature {}",
            (v - t[k]).abs()
        );
    }
}

#[test]
#[ignore = "GPU"]
fn g8_mutants() {
    let app = appareil();
    let mut non_detectes = vec![];
    // mutants de noyau 1, 2, 3, 5 ; mutant de signe = poids +1 dans le bras tordu (mutant = 0, poids_tordu = +1)
    let cas: Vec<(String, f64, u8)> = vec![
        ("signe (poids +1)".into(), 1.0, 0),
        ("noyau 1 (mauvais plan)".into(), -1.0, 1),
        ("noyau 2 (pile décalée)".into(), -1.0, 2),
        ("noyau 3 (z bas oublié)".into(), -1.0, 3),
        ("noyau 5 (z bas au coin s)".into(), -1.0, 5),
    ];
    for (nom, poids, mu) in cas {
        let (b, s, _) = direct_l2(&app, [2; 4], "23", 0.25, 8, 128, 300, 2000, poids, mu);
        let k = b.iter().position(|x| (x - 2.0).abs() < 1e-9).unwrap();
        let z = (s[k].0 + 1.25154) / (s[k].1 * s[k].1 + 0.01998f64.powi(2)).sqrt();
        println!(
            "G8 mutant {nom} : ln R(2) = {:+.4}({:.4}) z = {z:+.1}",
            s[k].0, s[k].1
        );
        if z.abs() <= 4.0 {
            non_detectes.push(nom);
        }
    }
    // le témoin sans mutant doit passer
    let (b, s, _) = direct_l2(&app, [2; 4], "23", 0.25, 8, 128, 300, 2000, -1.0, 0);
    let k = b.iter().position(|x| (x - 2.0).abs() < 1e-9).unwrap();
    let z = (s[k].0 + 1.25154) / (s[k].1 * s[k].1 + 0.01998f64.powi(2)).sqrt();
    println!(
        "G8 témoin sans mutant : ln R(2) = {:+.4}({:.4}) z = {z:+.1}",
        s[k].0, s[k].1
    );
    assert!(z.abs() <= 4.0, "le témoin sans mutant échoue");
    assert!(
        non_detectes.is_empty(),
        "mutants NON détectés : {non_detectes:?}"
    );
}

#[test]
#[ignore = "GPU"]
fn g6_beta_zero_et_g7_appariement() {
    let app = appareil();
    // G6 : β = 0, torsion plan 01, L = 4, 64 répliques
    let chs: Vec<ParamChaine> = (0..64).map(|i| ch(0.0, 900 + i, Depart::Chaud)).collect();
    let mut m = UneBoite::nouveau(&app, &pb([4; 4], "01", -1.0, chs)).unwrap();
    m.init().unwrap();
    let s = m.serie(50, 2000, 1).unwrap();
    let n = (64 * 2000) as f64;
    let mfe = s.iter().flatten().map(|x| x[1]).sum::<f64>() / n;
    let vfe = s
        .iter()
        .flatten()
        .map(|x| (x[1] - mfe).powi(2))
        .sum::<f64>()
        / n;
    let sig = (vfe / n).sqrt();
    println!("G6 : ⟨S_F⟩ = {mfe:+.5} ± {sig:.5} ; Var(S_F) = {vfe:.4} (att. 4,0)");
    assert!(mfe.abs() <= 4.0 * sig, "intégrande non nulle à β = 0");
    assert!((vfe / 4.0 - 1.0).abs() <= 0.15, "variance de S_F");
    // G7 : appariement
    let mk = |poids: f64| {
        let chs: Vec<ParamChaine> = (0..4).map(|i| ch(1.0, 50 + i, Depart::Froid)).collect();
        let mut m = UneBoite::nouveau(&app, &pb([2; 4], "23", poids, chs)).unwrap();
        m.init().unwrap();
        m.serie(10, 50, 1).unwrap()
    };
    let (a, b, c) = (mk(-1.0), mk(-1.0), mk(1.0));
    assert_eq!(a, b, "G7a : même lot rejoué : octets différents");
    let d = mk(1.0);
    assert_eq!(c, d, "G7b");
    // (c) covariance S_R(+) / S_R(−) : information
    let (mut sxy, mut sx, mut sy, mut n2) = (0.0, 0.0, 0.0, 0.0);
    for ch_ in 0..4 {
        for t in 0..50 {
            let (x, y) = (a[ch_][t][0] - a[ch_][t][1], c[ch_][t][0] - c[ch_][t][1]);
            sxy += x * y;
            sx += x;
            sy += y;
            n2 += 1.0;
        }
    }
    println!(
        "G7c : cov(S_R−, S_R+) = {:.4} (L=2, β=1, 200 mesures)",
        sxy / n2 - sx * sy / (n2 * n2)
    );
}

// ---------------------------------------------------------------- chemin rapide F32 (ADDENDUM 1)
#[test]
#[ignore = "GPU"]
fn f1_mesure_f32_egale_hote() {
    let app = appareil();
    for (cotes, tors, plans) in [
        ([4, 4, 4, 4], "01", vec![0usize]),
        ([2, 4, 6, 8], "02,13", vec![1, 4]),
    ] {
        let mut m = UneBoite::nouveau(
            &app,
            &pbp(
                cotes,
                tors,
                -1.0,
                vec![ch(2.3, 5, Depart::Froid)],
                Precision::F32,
            ),
        )
        .unwrap();
        m.init().unwrap();
        let q: Vec<f64> = champ_haar(cotes, 99)
            .iter()
            .map(|&x| f64::from(x as f32))
            .collect();
        m.televerser(0, &q).unwrap();
        m.allouer_historique(1).unwrap();
        m.mesurer(0).unwrap();
        let h = m.historique().unwrap();
        let o = [cotes[0] - 1, cotes[1] - 1, cotes[2] - 1, cotes[3] - 1];
        let tot = mesure_hote(cotes, &q, &[], o).0;
        let fe: f64 = plans
            .iter()
            .map(|&p| mesure_hote(cotes, &q, &[p], o).1)
            .sum();
        assert!(
            rel(h[0], tot) < 1e-11 && rel(h[1], fe) < 1e-11,
            "F1 {cotes:?} : {} {} contre {tot} {fe}",
            h[0],
            h[1]
        );
    }
    println!("PASS F1");
}

#[test]
#[ignore = "GPU"]
fn f2_loi_f32_contre_f64_8_230() {
    let app = appareil();
    let a = plaquette_repliques_p(
        &app,
        [8; 4],
        2.30,
        32,
        1500,
        2500,
        7_100_000,
        Precision::F32,
    );
    let b = plaquette_repliques_p(
        &app,
        [8; 4],
        2.30,
        32,
        1500,
        2500,
        7_100_000,
        Precision::F64,
    );
    println!(
        "F2 8⁴ β=2,3 : P32 − P64 = {:+.2e} ± {:.1e} (P32 = {:.7})",
        a.0 - b.0,
        (a.1 * a.1 + b.1 * b.1).sqrt(),
        a.0
    );
    assert!(
        (a.0 - b.0).abs() <= 4e-5,
        "écart F32/F64 hors du critère de loi large"
    );
}

fn ecrire_json(nom: &str, v: &str) {
    let d = std::path::PathBuf::from(std::env::var("KVU_F3_DIR").unwrap_or_else(|_| "/tmp".into()));
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join(nom), v).unwrap();
}

/// Une pièce de la porte F3 : (boîte, précision) : P moyen et son erreur entre répliques ; paramètres par l'environnement.
fn piece_f3(cotes: [usize; 4], prec: Precision, nom: &str) {
    let app = appareil();
    let r: usize = std::env::var("KVU_F3_REP")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(64);
    let nmes: usize = std::env::var("KVU_F3_NMES")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(2000);
    let g: u64 = std::env::var("KVU_F3_GRAINE")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(9_000_000);
    let (m, e, dh, edh, _) = plaquette_repliques_p(&app, cotes, 2.3, r, 1500, nmes, g, prec);
    println!("F3 pièce {nom} : P = {m:.8} ± {e:.2e} (R = {r}, nmes = {nmes}) ; moitiés {dh:+.1e} ± {edh:.1e}");
    ecrire_json(
        nom,
        &format!("{{\"p\": {m:.12}, \"e\": {e:.6e}, \"r\": {r}, \"nmes\": {nmes}}}"),
    );
}
#[test]
#[ignore = "GPU"]
fn f3_piece_488_f64() {
    piece_f3([4, 8, 8, 8], Precision::F64, "488_f64.json")
}
#[test]
#[ignore = "GPU"]
fn f3_piece_488_f32() {
    piece_f3([4, 8, 8, 8], Precision::F32, "488_f32.json")
}
#[test]
#[ignore = "GPU"]
fn f3_piece_8888_f64() {
    piece_f3([8, 8, 8, 8], Precision::F64, "8888_f64.json")
}
#[test]
#[ignore = "GPU"]
fn f3_piece_8888_f32() {
    piece_f3([8, 8, 8, 8], Precision::F32, "8888_f32.json")
}

#[test]
#[ignore = "GPU"]
fn f3_verdict() {
    let d = std::path::PathBuf::from(std::env::var("KVU_F3_DIR").unwrap_or_else(|_| "/tmp".into()));
    let lire = |n: &str| -> (f64, f64) {
        let t = std::fs::read_to_string(d.join(n))
            .unwrap_or_else(|_| panic!("pièce {n} absente : F3 NON CONCLUANTE"));
        let v: serde_json::Value = serde_json::from_str(&t).unwrap();
        (v["p"].as_f64().unwrap(), v["e"].as_f64().unwrap())
    };
    let diff = |a: &str, b: &str| {
        let (pa, ea) = lire(a);
        let (pb_, eb) = lire(b);
        (pa - pb_, (ea * ea + eb * eb).sqrt())
    };
    let (d64, e64) = diff("488_f64.json", "8888_f64.json");
    let (d32, e32) = diff("488_f32.json", "8888_f32.json");
    let z = (d32 - d64) / (e32 * e32 + e64 * e64).sqrt();
    println!("F3 : t(4888) − t(8888) : F64 {d64:+.7} ± {e64:.1e} ; F32 {d32:+.7} ± {e32:.1e} ; écart {:.2e} z = {z:+.2}", d32 - d64);
    let (a32, _) = lire("8888_f32.json");
    let (a64, _) = lire("8888_f64.json");
    println!(
        "F3 information : plaquette absolue 8⁴ F32 − F64 = {:+.2e}",
        a32 - a64
    );
    assert!(
        e32 <= 3e-7 && e64 <= 3e-7,
        "F3 NON CONCLUANTE : σ = {e32:.2e}, {e64:.2e} > 3e-7"
    );
    assert!(z.abs() <= 3.0, "F3 échoue : z = {z}");
}

#[test]
#[ignore = "GPU"]
fn f4_torsion_f32_l2() {
    let app = appareil();
    {
        let (poids, attendu_ok) = (-1.0, true);
        let betas: Vec<f64> = (1..=8).map(|k| 0.25 * k as f64).collect();
        let r = 128;
        let chs = |off: u64| -> Vec<ParamChaine> {
            betas
                .iter()
                .enumerate()
                .flat_map(|(k, &b)| {
                    (0..r).map(move |i| {
                        ch(
                            b,
                            off + (k * r + i) as u64,
                            if i % 2 == 0 {
                                Depart::Froid
                            } else {
                                Depart::Chaud
                            },
                        )
                    })
                })
                .collect()
        };
        let mut bras = vec![];
        for w in [poids, 1.0] {
            let mut m = UneBoite::nouveau(&app, &pbp([2; 4], "23", w, chs(41_000), Precision::F32))
                .unwrap();
            m.init().unwrap();
            bras.push(m.serie(300, 2000, 1).unwrap());
        }
        let pts = |b: usize| -> Vec<kvu::su2::thermo::PointRep> {
            let mut v = vec![];
            for (k, &be) in betas.iter().enumerate() {
                for i in 0..r {
                    v.push(kvu::su2::thermo::resumer(
                        be,
                        i,
                        0,
                        "x",
                        &bras[b][k * r + i],
                    ));
                }
            }
            v
        };
        let res = kvu::su2::thermo::ln_r(
            &pts(0),
            &pts(1),
            poids,
            kvu::su2::thermo::Quadrature::Parabole,
        )
        .unwrap();
        let k = betas.iter().position(|&b| b == 2.0).unwrap();
        let z = (res.valeur[k] + 1.25154) / (res.erreur[k].powi(2) + 0.01998f64.powi(2)).sqrt();
        println!(
            "F4 poids {poids} : ln R(2) F32 = {:+.4}({:.4}) z = {z:+.1}",
            res.valeur[k], res.erreur[k]
        );
        if attendu_ok {
            assert!(z.abs() <= 3.0);
        }
    }
}
