//! Tests CPU des observables (T7) : boucles de Wilson, Polyakov, flot RK3, trèfle, charge
//! topologique, t₀/w₀, grammaire de `--mesures` et écriture en ajout.
//!
//! Gates couvertes (ARCHITECTURE_UNIFIEE.md §6.4) : OBS-1 (champ froid, flux abélien, invariance de
//! jauge, accord avec une implémentation matricielle indépendante), OBS-3 (jauge pure, ordre RK3,
//! t₀ comme `wilson_flow_curve`). Les comparaisons au moteur historique (`avg_wilson_loop`,
//! `wilson_flow_step`) vivent dans `tests/unifie_obs_vs_historique.rs` (crate racine).

use kvu::commun::ChampHote;
use kvu::heritage::su2_hb::{measure_quat, qdag, qmul, quat_to_mat, CpuSu2, Geo, Q};
use kvu::obs::enligne::{self, ElementMesure};
use kvu::obs::flot::{echelles, flot_wilson, ParamsFlot};
use kvu::obs::ref_cpu::{self, orientations};
use kvu::obs::wilson::{boucles_wilson, correlateur_polyakov, ParamsWilson};
use std::f64::consts::PI;
use std::path::PathBuf;

// ------------------------------------------------------------------------------------------
// Outils de test : générateur indépendant, champs de référence
// ------------------------------------------------------------------------------------------

/// SplitMix64 : générateur déterministe, indépendant de celui du crate.
struct Alea(u64);
impl Alea {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniforme(&mut self) -> f64 {
        ((self.u64() >> 11) as f64 + 0.5) / 9007199254740992.0
    }
    fn normale(&mut self) -> f64 {
        (-2.0 * self.uniforme().ln()).sqrt() * (2.0 * PI * self.uniforme()).cos()
    }
    /// SU(2) de Haar : 4 gaussiennes normalisées.
    fn haar(&mut self) -> Q {
        let q = [self.normale(), self.normale(), self.normale(), self.normale()];
        let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
    /// SU(2) proche de l'identité : exp(ε · vecteur gaussien).
    fn proche_identite(&mut self, eps: f64) -> Q {
        ref_cpu::exp_su2([eps * self.normale(), eps * self.normale(), eps * self.normale()])
    }
}

/// Champ thermalisé (β = 2,3) par le générateur CPU de référence, `n_pas` pas (1 bain + 2 OR).
fn champ_thermalise_n(l: usize, graine: u64, n_pas: usize) -> ChampHote {
    let mut c = CpuSu2::new(l, graine).unwrap();
    c.init(true);
    for _ in 0..n_pas {
        c.step(2.3, 2);
    }
    ChampHote { l, q: c.flat() }
}

fn champ_thermalise(l: usize, graine: u64) -> ChampHote {
    champ_thermalise_n(l, graine, 6)
}

/// Champ de liens exp(ε ξ) tous indépendants : rugueux, mais proche de l'identité.
fn champ_proche_identite(l: usize, eps: f64, graine: u64) -> ChampHote {
    let mut a = Alea(graine);
    let v = l.pow(4);
    let mut q = Vec::with_capacity(16 * v);
    for _ in 0..4 * v {
        q.extend_from_slice(&a.proche_identite(eps));
    }
    ChampHote { l, q }
}

fn quats(c: &ChampHote) -> Vec<Q> {
    ref_cpu::en_quaternions(c)
}

/// Transformation de jauge U_μ(x) → g(x) U_μ(x) g(x+μ)†.
fn jauge(c: &ChampHote, g: &[Q]) -> ChampHote {
    let geo = Geo::new(c.l);
    let u = quats(c);
    let mut out = u.clone();
    for mu in 0..4 {
        for s in 0..geo.v {
            let x = geo.coords(s);
            let smu = geo.nb_up(s, mu, x[mu]);
            out[mu * geo.v + s] = qmul(qmul(g[s], u[mu * geo.v + s]), qdag(g[smu]));
        }
    }
    ref_cpu::en_champ(&out, c.l)
}

fn jauge_aleatoire(l: usize, graine: u64) -> Vec<Q> {
    let mut a = Alea(graine);
    (0..l.pow(4)).map(|_| a.haar()).collect()
}

fn ecart_max(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn params(r: usize, t: usize, toutes: bool) -> ParamsWilson {
    ParamsWilson {
        r_max: r,
        t_max: t,
        dir_temps: 3,
        toutes_orientations: toutes,
    }
}

/// Répertoire temporaire propre à un test, sous la cible de cargo (nettoyé par `cargo clean`).
fn dossier_test(nom: &str) -> PathBuf {
    let base = option_env!("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let d = base.join(format!("kvu_obs_{nom}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

// ------------------------------------------------------------------------------------------
// Champ abélien à flux constant : U(1) plongé dans SU(2), exp(iφσ3) = (cos φ, 0, 0, sin φ)
// ------------------------------------------------------------------------------------------

/// Flux constant θ = 2π n / L² dans le plan (μ, ν), puis flux θ2 dans le plan (μ2, ν2) (autres liens = 1).
/// Jauge : U_ν(x) = e^{iθ x_μ} ; U_μ(x) = e^{−iθ L x_ν} si x_μ = L−1, sinon 1.
fn champ_flux(l: usize, plans: &[((usize, usize), f64)]) -> ChampHote {
    let mut c = ChampHote::froid(l).unwrap();
    let geo = Geo::new(l);
    for &((mu, nu), theta) in plans {
        for s in 0..geo.v {
            let x = geo.coords(s);
            let phi_nu = theta * x[mu] as f64;
            let o = 4 * (nu * geo.v + s);
            c.q[o..o + 4].copy_from_slice(&[phi_nu.cos(), 0.0, 0.0, phi_nu.sin()]);
            if x[mu] == l - 1 {
                let phi_mu = -theta * l as f64 * x[nu] as f64;
                let o = 4 * (mu * geo.v + s);
                c.q[o..o + 4].copy_from_slice(&[phi_mu.cos(), 0.0, 0.0, phi_mu.sin()]);
            }
        }
    }
    c
}

// ------------------------------------------------------------------------------------------
// Implémentation matricielle INDÉPENDANTE (2×2 complexes) : boucle de Wilson
// ------------------------------------------------------------------------------------------

type M2 = [[(f64, f64); 2]; 2];

fn m2(q: Q) -> M2 {
    let m = quat_to_mat(q);
    [[(m[0], m[1]), (m[2], m[3])], [(m[4], m[5]), (m[6], m[7])]]
}
fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn mmul(a: &M2, b: &M2) -> M2 {
    let mut c = [[(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                let p = cmul(a[i][k], b[k][j]);
                c[i][j].0 += p.0;
                c[i][j].1 += p.1;
            }
        }
    }
    c
}
fn mdag(a: &M2) -> M2 {
    [
        [(a[0][0].0, -a[0][0].1), (a[1][0].0, -a[1][0].1)],
        [(a[0][1].0, -a[0][1].1), (a[1][1].0, -a[1][1].1)],
    ]
}

/// ½ Re Tr de la boucle R×T au site s, par produits matriciels, avec coordonnées explicites.
fn boucle_matricielle(c: &ChampHote, s: usize, mu: usize, nu: usize, r: usize, t: usize) -> f64 {
    let l = c.l;
    let geo = Geo::new(l);
    let u = quats(c);
    let lien = |d: usize, x: [usize; 4]| m2(u[d * geo.v + ((x[0] * l + x[1]) * l + x[2]) * l + x[3]]);
    let mut x = geo.coords(s);
    let mut p: M2 = [[(1.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)]];
    for _ in 0..r {
        p = mmul(&p, &lien(mu, x));
        x[mu] = (x[mu] + 1) % l;
    }
    for _ in 0..t {
        p = mmul(&p, &lien(nu, x));
        x[nu] = (x[nu] + 1) % l;
    }
    for _ in 0..r {
        x[mu] = (x[mu] + l - 1) % l;
        p = mmul(&p, &mdag(&lien(mu, x)));
    }
    for _ in 0..t {
        x[nu] = (x[nu] + l - 1) % l;
        p = mmul(&p, &mdag(&lien(nu, x)));
    }
    0.5 * (p[0][0].0 + p[1][1].0)
}

// ------------------------------------------------------------------------------------------
// OBS-1 : boucles de Wilson
// ------------------------------------------------------------------------------------------

#[test]
fn wilson_champ_froid_exactement_un() {
    let c = ChampHote::froid(4).unwrap();
    for toutes in [false, true] {
        let t = ref_cpu::table_wilson(&c, &params(4, 4, toutes)).unwrap();
        assert!(
            t.w.iter().all(|&w| w == 1.0),
            "W froid doit valoir 1 exactement : {:?}",
            t.w
        );
    }
    // via l'API publique (CPU, champ hôte seul)
    let t = boucles_wilson(None, None, 0, Some(&c), &params(3, 2, true)).unwrap();
    assert!(t.w.iter().all(|&w| w == 1.0));
    println!("PASS OBS-1 champ froid : W = 1 exactement");
}

#[test]
fn wilson_flux_abelien_constant_cos_rt_theta() {
    // Plan (μ,ν) = (0,1), un quantum de flux : θ = 2π / L².
    let l = 4;
    let theta = 2.0 * PI / (l * l) as f64;
    let c = champ_flux(l, &[((0, 1), theta)]);
    let mut pire: f64 = 0.0;
    for r in 1..=4 {
        for t in 1..=4 {
            let attendu = ((r * t) as f64 * theta).cos();
            for (mu, nu) in [(0, 1), (1, 0)] {
                let w = ref_cpu::wilson_orientation(&c, mu, nu, r, t).unwrap();
                pire = pire.max((w - attendu).abs());
            }
            // les autres plans ne voient aucun flux : W = 1
            for (mu, nu) in [
                (0, 2),
                (2, 0),
                (1, 3),
                (3, 1),
                (2, 3),
                (3, 2),
                (0, 3),
                (3, 0),
                (1, 2),
                (2, 1),
            ] {
                let w = ref_cpu::wilson_orientation(&c, mu, nu, r, t).unwrap();
                pire = pire.max((w - 1.0).abs());
            }
        }
    }
    assert!(pire <= 1e-13, "flux abélien : écart {pire:e}");
    // flux dans le plan (2,3) : la direction du temps 3 voit le flux avec R selon 2
    let c = champ_flux(l, &[((2, 3), theta)]);
    let t = ref_cpu::table_wilson(&c, &params(4, 4, false)).unwrap();
    // 3 orientations (0,3), (1,3), (2,3) : seule (2,3) a du flux → moyenne = (2 + cos)/3
    let mut pire: f64 = 0.0;
    for r in 1..=4 {
        for tt in 1..=4 {
            let attendu = (2.0 + ((r * tt) as f64 * theta).cos()) / 3.0;
            pire = pire.max((t.get(r, tt) - attendu).abs());
        }
    }
    assert!(pire <= 1e-13, "flux abélien (plan 2,3), moyenne temporelle : {pire:e}");
    println!("PASS OBS-1 flux abélien : cos(R·T·θ), écart max {pire:e}");
}

#[test]
fn wilson_invariance_de_jauge() {
    let c = champ_thermalise(4, 77);
    let g = jauge_aleatoire(4, 5);
    let cg = jauge(&c, &g);
    let p = params(4, 4, true);
    let a = ref_cpu::table_wilson(&c, &p).unwrap();
    let b = ref_cpu::table_wilson(&cg, &p).unwrap();
    let e = ecart_max(&a.w, &b.w);
    assert!(e <= 1e-13, "Wilson non invariant par jauge : {e:e}");
    // la configuration est bien non triviale
    assert!(
        a.get(1, 1) < 0.9 && a.get(4, 4).abs() < 0.5,
        "champ de test trivial : {:?}",
        a.w
    );
    // Polyakov : moyennes et corrélateur
    let (pa, pb) = (
        ref_cpu::polyakov_moyen(&c).unwrap(),
        ref_cpu::polyakov_moyen(&cg).unwrap(),
    );
    assert!(ecart_max(&pa, &pb) <= 1e-13);
    let (ca, cb) = (
        ref_cpu::correlateur_polyakov(&c, 3).unwrap(),
        ref_cpu::correlateur_polyakov(&cg, 3).unwrap(),
    );
    assert!(ecart_max(&ca, &cb) <= 1e-13, "corrélateur non invariant");
    println!("PASS OBS-1 invariance de jauge : Wilson {e:e}");
}

#[test]
fn wilson_quaternion_egale_matrices_complexes() {
    // implémentation matricielle indépendante, boucle par boucle, sur un champ thermalisé
    let c = champ_thermalise(4, 31);
    let geo = Geo::new(4);
    let mut pire: f64 = 0.0;
    for (mu, nu) in [(0, 1), (2, 3), (3, 0), (1, 2)] {
        for (r, t) in [(1, 1), (2, 1), (1, 3), (3, 2), (4, 4)] {
            let somme: f64 = (0..geo.v).map(|s| boucle_matricielle(&c, s, mu, nu, r, t)).sum();
            let moy = somme / geo.v as f64;
            let w = ref_cpu::wilson_orientation(&c, mu, nu, r, t).unwrap();
            pire = pire.max((w - moy).abs());
        }
    }
    assert!(pire <= 1e-13, "quaternion contre matrices 2×2 : {pire:e}");
    println!("PASS OBS-1 quaternion = matrices 2×2 complexes : {pire:e}");
}

#[test]
fn wilson_plaquette_egale_mesure_su2_hip() {
    // moyenne des 6 plans (μ<ν) de W(1,1) = plaquette de su2_hip (measure_quat)
    let c = champ_thermalise(4, 9);
    let geo = Geo::new(4);
    let ref64 = measure_quat(&geo, &quats(&c))[0];
    let somme: f64 = (0..4)
        .flat_map(|mu| (mu + 1..4).map(move |nu| (mu, nu)))
        .map(|(mu, nu)| ref_cpu::wilson_orientation(&c, mu, nu, 1, 1).unwrap())
        .sum();
    assert!(
        (somme / 6.0 - ref64).abs() <= 1e-15,
        "plaquette : {} contre {}",
        somme / 6.0,
        ref64
    );
    // E_plaq = 24 (1 − P)
    let d = ref_cpu::densites(&c).unwrap();
    assert!((d.e_plaq - 24.0 * (1.0 - ref64)).abs() <= 1e-13);
}

#[test]
fn wilson_orientations_et_dimensions() {
    let p = params(2, 2, false);
    assert_eq!(orientations(&p), vec![(0, 3), (1, 3), (2, 3)]);
    let p = ParamsWilson { dir_temps: 1, ..p };
    assert_eq!(orientations(&p), vec![(0, 1), (2, 1), (3, 1)]);
    assert_eq!(orientations(&params(2, 2, true)).len(), 12);
    // paramètres invalides
    let c = ChampHote::froid(4).unwrap();
    assert!(ref_cpu::table_wilson(&c, &params(0, 2, false)).is_err());
    assert!(ref_cpu::table_wilson(
        &c,
        &ParamsWilson {
            dir_temps: 4,
            ..params(1, 1, false)
        }
    )
    .is_err());
    assert!(ref_cpu::wilson_orientation(&c, 1, 1, 1, 1).is_err());
    // sans champ : erreur de configuration
    assert!(boucles_wilson(None, None, 0, None, &params(1, 1, false)).is_err());
}

#[test]
fn wilson_table_egale_moyenne_des_orientations() {
    let c = champ_thermalise(4, 4);
    let p = params(3, 2, true);
    let t = ref_cpu::table_wilson(&c, &p).unwrap();
    for r in 1..=3 {
        for tt in 1..=2 {
            let somme: f64 = orientations(&p)
                .iter()
                .map(|&(mu, nu)| ref_cpu::wilson_orientation(&c, mu, nu, r, tt).unwrap())
                .sum();
            assert!((t.get(r, tt) - somme / 12.0).abs() <= 1e-15);
        }
    }
}

// ------------------------------------------------------------------------------------------
// OBS-2 (volet CPU) : Polyakov
// ------------------------------------------------------------------------------------------

#[test]
fn polyakov_egale_mesure_ref64() {
    let c = champ_thermalise(6, 3);
    let geo = Geo::new(6);
    let ref64 = measure_quat(&geo, &quats(&c));
    let p = ref_cpu::polyakov_moyen(&c).unwrap();
    for mu in 0..4 {
        assert!(
            (p[mu] - ref64[1 + mu]).abs() <= 1e-15,
            "P{mu} : {} contre {}",
            p[mu],
            ref64[1 + mu]
        );
    }
}

#[test]
fn polyakov_champ_froid_et_correlateur() {
    let c = ChampHote::froid(6).unwrap();
    let corr = correlateur_polyakov(None, None, 0, Some(&c), 3).unwrap();
    assert_eq!(corr.len(), 4);
    assert!(corr.iter().all(|&x| (x - 1.0).abs() <= 1e-15));
    // C(0) = ⟨ℓ²⟩ ≥ ⟨ℓ⟩² (Cauchy-Schwarz) sur une configuration quelconque
    let c = champ_thermalise(6, 12);
    let corr = ref_cpu::correlateur_polyakov(&c, 3).unwrap();
    let p3 = ref_cpu::polyakov_moyen(&c).unwrap()[3];
    assert!(corr[0] >= p3 * p3 - 1e-15);
    assert!(corr[0] <= 1.0 + 1e-15);
}

#[test]
fn polyakov_correlateur_contre_calcul_direct() {
    // calcul naïf indépendant : ℓ(x) par produit le long de la direction 3, corrélateur par triple boucle
    let l = 4;
    let c = champ_thermalise(l, 21);
    let u = quats(&c);
    let geo = Geo::new(l);
    let ell = |x0: usize, x1: usize, x2: usize| -> f64 {
        let mut p: Q = [1.0, 0.0, 0.0, 0.0];
        for x3 in 0..l {
            p = qmul(p, u[3 * geo.v + ((x0 * l + x1) * l + x2) * l + x3]);
        }
        p[0]
    };
    let corr = ref_cpu::correlateur_polyakov(&c, 3).unwrap();
    for (r, &c_r) in corr.iter().enumerate() {
        let mut somme = 0.0;
        for x0 in 0..l {
            for x1 in 0..l {
                for x2 in 0..l {
                    let e = ell(x0, x1, x2);
                    somme +=
                        e * ell((x0 + r) % l, x1, x2) + e * ell(x0, (x1 + r) % l, x2) + e * ell(x0, x1, (x2 + r) % l);
                }
            }
        }
        let attendu = somme / (3.0 * (l * l * l) as f64);
        assert!((c_r - attendu).abs() <= 1e-14, "C({r}) : {c_r} contre {attendu}");
    }
}

// ------------------------------------------------------------------------------------------
// OBS-3 : flot
// ------------------------------------------------------------------------------------------

#[test]
fn flot_champ_froid_reste_froid() {
    let c = ChampHote::froid(4).unwrap();
    let p = ParamsFlot {
        dt: 0.05,
        t_max: 0.5,
        ..ParamsFlot::default()
    };
    let courbe = flot_wilson(None, None, 0, Some(&c), &p).unwrap();
    assert_eq!(courbe.t.len(), 11);
    assert!(courbe.e_plaq.iter().all(|&e| e == 0.0));
    assert!(courbe.e_trefle.iter().all(|&e| e == 0.0));
    assert!(courbe.q_top.iter().all(|&q| q == 0.0));
    assert_eq!(courbe.t0, None);
    assert_eq!(courbe.w0, None);
}

#[test]
fn flot_jauge_pure_energie_nulle() {
    // U_μ(x) = g(x) g(x+μ)† : toutes les plaquettes valent 1, E ≡ 0 le long du flot
    let l = 4;
    let froid = ChampHote::froid(l).unwrap();
    let g = jauge_aleatoire(l, 8);
    let pur = jauge(&froid, &g);
    let p = ParamsFlot {
        dt: 0.02,
        t_max: 0.2,
        mesure_chaque: 2,
        ..ParamsFlot::default()
    };
    let courbe = flot_wilson(None, None, 0, Some(&pur), &p).unwrap();
    let pire = courbe
        .e_plaq
        .iter()
        .chain(&courbe.e_trefle)
        .chain(&courbe.q_top)
        .fold(0.0f64, |m, x| m.max(x.abs()));
    assert!(pire <= 1e-14, "jauge pure : E, Q ≠ 0 de {pire:e}");
    println!("PASS OBS-3 jauge pure : E ≡ 0 (max {pire:e})");
}

#[test]
fn flot_invariance_de_jauge() {
    let c = champ_thermalise(4, 14);
    let cg = jauge(&c, &jauge_aleatoire(4, 15));
    let p = ParamsFlot {
        dt: 0.02,
        t_max: 0.1,
        ..ParamsFlot::default()
    };
    let a = ref_cpu::flot(&c, &p).unwrap();
    let b = ref_cpu::flot(&cg, &p).unwrap();
    let e = ecart_max(&a.e_plaq, &b.e_plaq)
        .max(ecart_max(&a.e_trefle, &b.e_trefle))
        .max(ecart_max(&a.q_top, &b.q_top));
    assert!(e <= 1e-11, "flot non covariant : {e:e}");
    assert!(a.e_plaq[0] > 1.0, "champ trivial");
    // E décroît strictement le long du flot (gradient descendant de l'action)
    assert!(
        a.e_plaq.windows(2).all(|w| w[1] < w[0]),
        "E_plaq doit décroître : {:?}",
        a.e_plaq
    );
}

#[test]
fn flot_ordre_trois_de_rk3() {
    // champ lisse (exp(0,3 ξ) par lien) ; erreur sur les liens à t = 0,08 pour dt et dt/2,
    // contre une solution de référence à dt/16
    let c = champ_proche_identite(4, 0.3, 99);
    let t_fin = 0.08;
    let ref_fin = ref_cpu::flot_n_pas(&c, t_fin / 64.0, 64).unwrap();
    let erreur = |n: usize| {
        let f = ref_cpu::flot_n_pas(&c, t_fin / n as f64, n).unwrap();
        ecart_max(&f.q, &ref_fin.q)
    };
    let (e1, e2, e4) = (erreur(2), erreur(4), erreur(8));
    let rapport = e1 / e2;
    let rapport2 = e2 / e4;
    println!("ordre RK3 : erreurs {e1:e} {e2:e} {e4:e}, rapports {rapport:.3} {rapport2:.3}");
    assert!((6.0..=10.0).contains(&rapport), "rapport dt / dt/2 = {rapport}");
    assert!((6.0..=10.0).contains(&rapport2), "rapport dt/2 / dt/4 = {rapport2}");
}

#[test]
fn flot_unitarite_conservee() {
    let c = champ_thermalise(4, 22);
    let f = ref_cpu::flot_n_pas(&c, 0.05, 20).unwrap();
    assert!(f.defaut_unitarite() <= 1e-14, "unitarité : {:e}", f.defaut_unitarite());
}

#[test]
fn flot_ne_modifie_pas_le_champ_source() {
    let c = champ_thermalise(4, 23);
    let avant = c.clone();
    let p = ParamsFlot {
        dt: 0.05,
        t_max: 0.2,
        ..ParamsFlot::default()
    };
    let _ = flot_wilson(None, None, 0, Some(&c), &p).unwrap();
    let _ = ref_cpu::flot_n_pas(&c, 0.05, 3).unwrap();
    assert_eq!(c, avant);
}

#[test]
fn flot_parametres_invalides() {
    let c = ChampHote::froid(4).unwrap();
    for p in [
        ParamsFlot {
            dt: 0.0,
            ..ParamsFlot::default()
        },
        ParamsFlot {
            dt: -0.1,
            ..ParamsFlot::default()
        },
        ParamsFlot {
            t_max: 0.0,
            ..ParamsFlot::default()
        },
        ParamsFlot {
            dt: 1.0,
            t_max: 0.5,
            ..ParamsFlot::default()
        },
        ParamsFlot {
            mesure_chaque: 0,
            ..ParamsFlot::default()
        },
        ParamsFlot {
            dt: f64::NAN,
            ..ParamsFlot::default()
        },
    ] {
        assert!(
            flot_wilson(None, None, 0, Some(&c), &p).is_err(),
            "{p:?} aurait dû être refusé"
        );
    }
    assert!(flot_wilson(None, None, 0, None, &ParamsFlot::default()).is_err());
}

#[test]
fn flot_pas_de_mesure_et_dernier_point() {
    let c = champ_thermalise(4, 24);
    // 10 pas, mesure tous les 4 : pas 0, 4, 8 et le dernier (10)
    let p = ParamsFlot {
        dt: 0.02,
        t_max: 0.2,
        mesure_chaque: 4,
        ..ParamsFlot::default()
    };
    let courbe = ref_cpu::flot(&c, &p).unwrap();
    let t: Vec<f64> = courbe.t.iter().map(|x| (x * 1000.0).round() / 1000.0).collect();
    assert_eq!(t, vec![0.0, 0.08, 0.16, 0.2]);
    // même courbe aux points communs qu'avec une mesure à chaque pas
    let p1 = ParamsFlot {
        mesure_chaque: 1,
        ..p.clone()
    };
    let fine = ref_cpu::flot(&c, &p1).unwrap();
    for (i, k) in [0usize, 4, 8, 10].iter().enumerate() {
        assert_eq!(courbe.e_plaq[i], fine.e_plaq[*k]);
        assert_eq!(courbe.q_top[i], fine.q_top[*k]);
    }
}

// ------------------------------------------------------------------------------------------
// Trèfle et charge topologique
// ------------------------------------------------------------------------------------------

#[test]
fn trefle_flux_constant_formules_analytiques() {
    // un plan : E_plaq = 4 (1 − cos θ), E_trèfle = 2 sin²θ, Q = 0
    let l = 4;
    let theta = 2.0 * PI / (l * l) as f64;
    let c = champ_flux(l, &[((0, 1), theta)]);
    let d = ref_cpu::densites(&c).unwrap();
    assert!(
        (d.e_plaq - 4.0 * (1.0 - theta.cos())).abs() <= 1e-13,
        "E_plaq {} ",
        d.e_plaq
    );
    assert!(
        (d.e_trefle - 2.0 * theta.sin().powi(2)).abs() <= 1e-13,
        "E_trefle {}",
        d.e_trefle
    );
    assert!(d.q_top.abs() <= 1e-13);
}

#[test]
fn topologie_flux_croises_normalisation() {
    // flux θ1 dans (0,1) et θ2 dans (2,3) : q = sinθ1 sinθ2 / (2π²) par site, Q = V q.
    // Limite continue : Q = 2 n1 n2 (θ = 2π n / L²).
    let l = 8;
    let theta = 2.0 * PI / (l * l) as f64;
    let c = champ_flux(l, &[((0, 1), theta), ((2, 3), theta)]);
    let d = ref_cpu::densites(&c).unwrap();
    let v = (l * l * l * l) as f64;
    let attendu = v * theta.sin() * theta.sin() / (2.0 * PI * PI);
    assert!((d.q_top - attendu).abs() <= 1e-12, "Q = {} contre {attendu}", d.q_top);
    assert!(
        (d.q_top - 2.0).abs() < 2e-2,
        "Q doit tendre vers 2 n1 n2 = 2 : {}",
        d.q_top
    );
    // signes : flux opposé dans le second plan ⟹ Q → −Q
    let c2 = champ_flux(l, &[((0, 1), theta), ((2, 3), -theta)]);
    let d2 = ref_cpu::densites(&c2).unwrap();
    assert!((d2.q_top + d.q_top).abs() <= 1e-12);
    // plans (0,2) et (1,3) : ε_{0213} = −1 ⟹ Q → −2
    let c3 = champ_flux(l, &[((0, 2), theta), ((1, 3), theta)]);
    let d3 = ref_cpu::densites(&c3).unwrap();
    assert!((d3.q_top + attendu).abs() <= 1e-12, "plans (0,2),(1,3) : {} ", d3.q_top);
    // plans (0,3) et (1,2) : ε_{0312} = +1
    let c4 = champ_flux(l, &[((0, 3), theta), ((1, 2), theta)]);
    let d4 = ref_cpu::densites(&c4).unwrap();
    assert!((d4.q_top - attendu).abs() <= 1e-12, "plans (0,3),(1,2) : {} ", d4.q_top);
    println!("PASS topologie : Q = {:.6} (continu 2), exact {attendu:.6}", d.q_top);
}

#[test]
fn topologie_invariance_de_jauge_et_parite() {
    let l = 4;
    let c = champ_thermalise(l, 55);
    let d = ref_cpu::densites(&c).unwrap();
    let dg = ref_cpu::densites(&jauge(&c, &jauge_aleatoire(l, 56))).unwrap();
    assert!((d.q_top - dg.q_top).abs() <= 1e-12, "Q non invariant par jauge");
    assert!((d.e_trefle - dg.e_trefle).abs() <= 1e-12);
    assert!((d.e_plaq - dg.e_plaq).abs() <= 1e-12);
    // réflexion x0 → −x0 : U'_0(x) = U_0(Rx − e_0)†, U'_i(x) = U_i(Rx) ; Q → −Q
    let geo = Geo::new(l);
    let u = quats(&c);
    let refl = |s: usize| -> usize {
        let mut x = geo.coords(s);
        x[0] = (l - x[0]) % l;
        ((x[0] * l + x[1]) * l + x[2]) * l + x[3]
    };
    let mut r = u.clone();
    for s in 0..geo.v {
        let mut x = geo.coords(s);
        let rs = refl(s);
        x[0] = (l - x[0] + l - 1) % l; // (−x0 − 1) mod L
        let s_m = ((x[0] * l + x[1]) * l + x[2]) * l + x[3];
        r[s] = qdag(u[s_m]);
        for mu in 1..4 {
            r[mu * geo.v + s] = u[mu * geo.v + rs];
        }
    }
    let dr = ref_cpu::densites(&ref_cpu::en_champ(&r, l)).unwrap();
    assert!(
        (dr.q_top + d.q_top).abs() <= 1e-12,
        "parité : Q = {} puis {}",
        d.q_top,
        dr.q_top
    );
    assert!((dr.e_trefle - d.e_trefle).abs() <= 1e-12);
    assert!(
        d.q_top.abs() > 1e-4,
        "champ de test à Q ≈ 0 : le test de parité serait vide"
    );
}

#[test]
fn trefle_proche_plaquette_en_champ_lisse() {
    // champ lisse : flux constant faible. Trèfle et plaquette concordent à O(θ⁴).
    let l = 8;
    let theta = 2.0 * PI / (l * l) as f64;
    let c = champ_flux(l, &[((1, 2), theta)]);
    let d = ref_cpu::densites(&c).unwrap();
    // E_trèfle/E_plaq = (1 + cos θ)/2 exactement, soit 1 − θ²/4 : écart relatif < θ²/3
    assert!(
        (d.e_trefle / d.e_plaq - 1.0).abs() < theta * theta / 3.0,
        "{} contre {}",
        d.e_trefle,
        d.e_plaq
    );
    assert!((d.e_trefle / d.e_plaq - (1.0 + theta.cos()) / 2.0).abs() <= 1e-11);
    // 1 − P perd ~3 chiffres par annulation
}

// ------------------------------------------------------------------------------------------
// t0 et w0
// ------------------------------------------------------------------------------------------

/// Transcription indépendante de `wilson_flow_curve` (src/observables/wilson_flow.rs) opérant sur une
/// courbe (t, t²E) à pas régulier : t0/a² et w0/a.
fn oracle_historique(t2e: &[f64], dtau: f64) -> (f64, f64) {
    let n = t2e.len();
    let t: Vec<f64> = (0..n).map(|k| k as f64 * dtau).collect();
    let mut w = vec![0.0; n];
    if n >= 3 {
        for i in 1..n - 1 {
            let d = (t2e[i + 1] - t2e[i - 1]) / (2.0 * dtau);
            w[i] = t[i] * d;
        }
        w[0] = t[0] * ((t2e[1] - t2e[0]) / dtau);
        w[n - 1] = t[n - 1] * ((t2e[n - 1] - t2e[n - 2]) / dtau);
    }
    let interp = |tl: f64, fl: f64, th: f64, fh: f64, cible: f64| {
        let den = fh - fl;
        if den.abs() < 1e-300 {
            tl
        } else {
            tl + (cible - fl) / den * (th - tl)
        }
    };
    let mut t0 = 0.0;
    for i in 1..n {
        if t2e[i - 1] < 0.3 && t2e[i] >= 0.3 {
            t0 = interp(t[i - 1], t2e[i - 1], t[i], t2e[i], 0.3);
            break;
        }
    }
    let mut w0 = 0.0;
    for i in 1..n {
        if w[i - 1] < 0.3 && w[i] >= 0.3 {
            let tc = interp(t[i - 1], w[i - 1], t[i], w[i], 0.3);
            w0 = if tc > 0.0 { tc.sqrt() } else { 0.0 };
            break;
        }
    }
    (t0, w0)
}

#[test]
fn echelles_courbe_synthetique_et_oracle() {
    // t²E = 0,15 t exact : t0 = 2 ; W = t · 0,15 → w0 = √2
    let dt = 0.7;
    let t: Vec<f64> = (0..8).map(|k| k as f64 * dt).collect();
    let t2e: Vec<f64> = t.iter().map(|x| 0.15 * x).collect();
    let (t0, w0) = echelles(&t, &t2e, 0.3, 0.3);
    assert!((t0.unwrap() - 2.0).abs() <= 1e-12);
    assert!((w0.unwrap() - 2.0f64.sqrt()).abs() <= 1e-12);
    let (o0, ow) = oracle_historique(&t2e, dt);
    assert!((t0.unwrap() - o0).abs() <= 1e-12 && (w0.unwrap() - ow).abs() <= 1e-12);
    // non encadré : None
    assert_eq!(echelles(&t, &t2e, 5.0, 5.0), (None, None));
    assert_eq!(echelles(&[0.0], &[0.0], 0.3, 0.3), (None, None));
    // cibles personnalisées
    let (t0b, _) = echelles(&t, &t2e, 0.45, 0.3);
    assert!((t0b.unwrap() - 3.0).abs() <= 1e-12);
}

#[test]
fn t0_w0_d_un_vrai_flot_egaux_a_l_oracle_historique() {
    // L = 6, β = 2,3, 30 pas : t²E traverse 0,3 vers t ≈ 1,1 (L = 4 flotte trop vite vers le trivial)
    let c = champ_thermalise_n(6, 101, 30);
    let dt = 0.05;
    let p = ParamsFlot {
        dt,
        t_max: 2.0,
        mesure_chaque: 1,
        cible_t2e: 0.3,
        cible_w: 0.3,
    };
    let courbe = ref_cpu::flot(&c, &p).unwrap();
    let (o0, ow) = oracle_historique(&courbe.t2e, dt);
    let t0 = courbe.t0.expect("t0 non encadré : courbe inattendue");
    assert!((t0 - o0).abs() <= 1e-12, "t0 {t0} contre {o0}");
    let w0 = courbe.w0.expect("w0 non encadré");
    assert!((w0 - ow).abs() <= 1e-12, "w0 {w0} contre {ow}");
    // t²E croît et traverse 0,3
    assert!(courbe.t2e.windows(2).take(10).all(|w| w[1] > w[0]));
    println!("PASS OBS-3 t0 = {t0:.6}, w0 = {w0:.6} (oracle {o0:.6}, {ow:.6})");
}

// ------------------------------------------------------------------------------------------
// Grammaire de --mesures (golden)
// ------------------------------------------------------------------------------------------

#[test]
fn grammaire_mesures_golden() {
    let el = enligne::decrire("wilson:R4xT4:10, polyakov_corr:10 ,flot:0.01:3.0:20").unwrap();
    assert_eq!(
        el,
        vec![
            ElementMesure::Wilson {
                params: ParamsWilson {
                    r_max: 4,
                    t_max: 4,
                    dir_temps: 3,
                    toutes_orientations: false
                },
                periode: 10
            },
            ElementMesure::PolyakovCorr {
                dir_temps: 3,
                periode: 10
            },
            ElementMesure::Flot {
                params: ParamsFlot {
                    dt: 0.01,
                    t_max: 3.0,
                    mesure_chaque: 1,
                    cible_t2e: 0.3,
                    cible_w: 0.3
                },
                periode: 20
            },
        ]
    );
    let el = enligne::decrire("wilson:R8xT6:toutes:5,flot:0.02:2:7:cible=0.4:chaque=5").unwrap();
    assert_eq!(
        el,
        vec![
            ElementMesure::Wilson {
                params: ParamsWilson {
                    r_max: 8,
                    t_max: 6,
                    dir_temps: 3,
                    toutes_orientations: true
                },
                periode: 5
            },
            ElementMesure::Flot {
                params: ParamsFlot {
                    dt: 0.02,
                    t_max: 2.0,
                    mesure_chaque: 5,
                    cible_t2e: 0.4,
                    cible_w: 0.4
                },
                periode: 7
            },
        ]
    );
    let el = enligne::decrire("flot:0.01:1:3:cible_t2e=0.3:cible_w=0.25").unwrap();
    assert_eq!(
        el,
        vec![ElementMesure::Flot {
            params: ParamsFlot {
                dt: 0.01,
                t_max: 1.0,
                mesure_chaque: 1,
                cible_t2e: 0.3,
                cible_w: 0.25
            },
            periode: 3
        }]
    );
    assert!(enligne::decrire("").unwrap().is_empty());
    assert_eq!(
        enligne::decrire("wilson:r2xt3:0").unwrap().len(),
        1,
        "période 0 = jamais, minuscules acceptées"
    );

    // les mesures construites portent les bons noms et périodes
    let ms = enligne::analyser_avec("wilson:R2xT2:10,polyakov_corr:5,flot:0.05:0.5:20", None).unwrap();
    let noms: Vec<(&str, usize)> = ms.iter().map(|m| (m.nom(), m.periode())).collect();
    assert_eq!(noms, vec![("wilson", 10), ("polyakov_corr", 5), ("flot", 20)]);
}

#[test]
fn grammaire_mesures_erreurs() {
    for mauvais in [
        "wilson",
        "wilson:R4xT4",
        "wilson:R4:10",
        "wilson:R0xT4:10",
        "wilson:R4xT4:toutes",
        "wilson:R4xT4:x",
        "wilson:R4xT4:10:11",
        "wilson:R4xT4:10,wilson:R2xT2:5",
        "polyakov_corr",
        "polyakov_corr:a",
        "polyakov_corr:5:6",
        "flot:0.01:3.0",
        "flot:0:3:10",
        "flot:0.5:0.1:10",
        "flot:0.01:3:10:cible",
        "flot:0.01:3:10:cible=abc",
        "flot:0.01:3:10:chaque=0",
        "flot:0.01:3:10:bidule=1",
        "plaquette:5",
    ] {
        let e = enligne::decrire(mauvais)
            .err()
            .unwrap_or_else(|| panic!("« {mauvais} » aurait dû être refusé"));
        assert_eq!(e.code_sortie(), 2, "erreur d'usage attendue pour « {mauvais} »");
        assert!(e.to_string().contains("--mesures"), "message en français attendu : {e}");
    }
}

// ------------------------------------------------------------------------------------------
// Mesures en ligne : écriture en ajout
// ------------------------------------------------------------------------------------------

fn lire(chemin: PathBuf) -> Vec<String> {
    std::fs::read_to_string(&chemin)
        .unwrap_or_else(|e| panic!("{} : {e}", chemin.display()))
        .lines()
        .map(String::from)
        .collect()
}

#[test]
fn mesures_en_ligne_ecritures_en_ajout() {
    let dossier = dossier_test("enligne");
    let sortie = dossier.join("c0");
    let c = champ_thermalise(4, 61);
    let mut ms = enligne::analyser_avec("wilson:R2xT3:10,polyakov_corr:5,flot:0.05:0.2:20", None).unwrap();
    for pas in [10u64, 20] {
        for m in ms.iter_mut() {
            m.mesurer(None, None, Some(&c), 0, pas, &sortie).unwrap();
        }
    }
    // wilson.csv : un seul en-tête, 2 mesures × 6 lignes
    let w = lire(sortie.join("wilson.csv"));
    assert_eq!(w[0], "pas,r,t,W");
    assert_eq!(w.len(), 1 + 2 * 6);
    assert!(w[1].starts_with("10,1,1,") && w[7].starts_with("20,1,1,"));
    let t = ref_cpu::table_wilson(&c, &params(2, 3, false)).unwrap();
    let v: f64 = w[2].rsplit(',').next().unwrap().parse().unwrap(); // r = 1, t = 2
    assert_eq!(v, t.get(1, 2), "la valeur écrite doit se relire exactement (%.17g)");
    // polyakov_corr.csv : L/2 + 1 = 3 lignes par mesure
    let p = lire(sortie.join("polyakov_corr.csv"));
    assert_eq!(p[0], "pas,r,C");
    assert_eq!(p.len(), 1 + 2 * 3);
    // flot.csv : 5 échantillons (t = 0, …, 0,2), deux mesures ; flot_t0.csv : une ligne par mesure
    let f = lire(sortie.join("flot.csv"));
    assert_eq!(f[0], "pas,t,E_plaq,E_trefle,t2E,Q");
    assert_eq!(f.len(), 1 + 2 * 5);
    let t0 = lire(sortie.join("flot_t0.csv"));
    assert_eq!(t0[0], "pas,t0,w0");
    assert_eq!(t0.len(), 3);
    assert!(t0[1].starts_with("10,") && t0[2].starts_with("20,"));
    assert!(
        t0[1].ends_with("nan,nan"),
        "flot court : t0 et w0 non encadrés : {}",
        t0[1]
    );
    // un second appel dans un nouveau processus ajoute sans réécrire l'en-tête (fichier existant)
    let mut ms2 = enligne::analyser_avec("wilson:R2xT3:10", None).unwrap();
    ms2[0].mesurer(None, None, Some(&c), 0, 30, &sortie).unwrap();
    let w2 = lire(sortie.join("wilson.csv"));
    assert_eq!(w2.len(), 1 + 3 * 6);
    assert_eq!(w2.iter().filter(|l| l.starts_with("pas")).count(), 1);
}

#[test]
fn mesures_en_ligne_sans_champ_est_une_erreur() {
    let dossier = dossier_test("sans_champ");
    let mut ms = enligne::analyser_avec("wilson:R1xT1:1", None).unwrap();
    assert!(ms[0].mesurer(None, None, None, 0, 1, &dossier).is_err());
    assert!(
        !dossier.join("wilson.csv").exists(),
        "aucun fichier ne doit naître d'une mesure en échec"
    );
}

#[test]
fn mesures_en_ligne_garde_ssd_refus_avant_ecriture() {
    use kvu::stockage::GardeSsd;
    // seuil absurde : refus AVANT tout octet écrit (rc 7) ; tant que T8 n'est pas fusionné, la garde
    // rend « non implémenté » et l'écriture est tolérée (STUB-T8) : on ne juge alors que l'absence de panique.
    let dossier = dossier_test("garde");
    let c = ChampHote::froid(4).unwrap();
    let mut ms = enligne::analyser_avec("wilson:R1xT1:1", Some(GardeSsd::nouvelle(1e12))).unwrap();
    match ms[0].mesurer(None, None, Some(&c), 0, 1, &dossier) {
        Err(e) => {
            assert_eq!(e.code_sortie(), 7, "refus SSD attendu : {e}");
            assert!(!dossier.join("wilson.csv").exists());
        }
        Ok(()) => println!("SKIP garde SSD : GardeSsd::verifier pas encore livré (T8)"),
    }
}

// ------------------------------------------------------------------------------------------
// Régression : valeurs gelées d'un champ déterministe (L = 4, graine 2709, 6 pas à β = 2,3)
// ------------------------------------------------------------------------------------------

#[test]
fn regression_valeurs_gelees() {
    let c = champ_thermalise(4, 2709);
    let p = params(2, 2, true);
    let w = ref_cpu::table_wilson(&c, &p).unwrap();
    let corr = ref_cpu::correlateur_polyakov(&c, 3).unwrap();
    let fl = ref_cpu::flot(
        &c,
        &ParamsFlot {
            dt: 0.05,
            t_max: 0.25,
            mesure_chaque: 5,
            ..ParamsFlot::default()
        },
    )
    .unwrap();
    let observe = serde_json::json!({
        "W": w.w, "C": corr, "E_plaq": fl.e_plaq, "E_trefle": fl.e_trefle, "Q": fl.q_top,
    });
    let chemin = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/donnees/obs/golden_L4_s2709.json");
    if std::env::var_os("KVU_OBS_REGEN").is_some() {
        std::fs::write(&chemin, serde_json::to_string_pretty(&observe).unwrap() + "\n").unwrap();
        println!("golden régénéré : {}", chemin.display());
        return;
    }
    let attendu: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&chemin).expect("golden absent")).unwrap();
    for cle in ["W", "C", "E_plaq", "E_trefle", "Q"] {
        let a: Vec<f64> = attendu[cle]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let b: Vec<f64> = observe[cle]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        // tolérance 1e-10 : les transcendantes de la libm hôte (log, cos) varient à l'ULP selon la machine
        assert!(ecart_max(&a, &b) <= 1e-10, "{cle} : écart {:e}", ecart_max(&a, &b));
    }
}
