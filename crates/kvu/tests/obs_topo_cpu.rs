//! Gates topologiques CPU : références matricielles et aléa de test indépendants.

use kvu::commun::{ChampHote, Erreur};
use kvu::heritage::su2_hb::{qnorm, quat_to_mat, CpuSu2, Geo, Q};
use kvu::obs::flot::echelles;
use kvu::obs::ref_cpu::{self, en_champ, en_quaternions, Reseau, PLANS, UN};
use kvu::obs::topo::*;
use std::f64::consts::TAU;
use std::sync::OnceLock;

/// SplitMix64, indépendant du Philox instrumental et du générateur de production.
struct Alea(u64);
impl Alea {
    fn entier(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniforme(&mut self) -> f64 {
        ((self.entier() >> 12) as f64 + 0.5) / 4503599627370496.0
    }
    fn normale(&mut self) -> f64 {
        (-2.0 * self.uniforme().ln()).sqrt() * (TAU * self.uniforme()).cos()
    }
    fn haar(&mut self) -> Q {
        let mut q = [
            self.normale(),
            self.normale(),
            self.normale(),
            self.normale(),
        ];
        let n = qnorm(q);
        for x in &mut q {
            *x /= n;
        }
        q
    }
}

fn thermalise(l: usize, beta: f64, graine: u64) -> ChampHote {
    let mut cpu = CpuSu2::new(l, graine).unwrap();
    cpu.init(true);
    for _ in 0..24 {
        cpu.step(beta, 2);
    }
    en_champ(&cpu.u, l)
}

fn champ4() -> &'static ChampHote {
    static CHAMP: OnceLock<ChampHote> = OnceLock::new();
    CHAMP.get_or_init(|| thermalise(4, 2.5, 26102619003))
}

fn record4() -> &'static RecordTopo {
    static RECORD: OnceLock<RecordTopo> = OnceLock::new();
    RECORD.get_or_init(|| {
        mesurer_cpu(
            champ4(),
            &ParamsTopo {
                pas_max: 500,
                ..ParamsTopo::default()
            },
        )
        .unwrap()
    })
}

// Matrices complexes 2×2 : (réel, imaginaire), rangées par ligne.
type C = [f64; 2];
type M = [[C; 2]; 2];
const IDENTITE: M = [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]];

fn cmul(a: C, b: C) -> C {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

fn matrice(q: Q) -> M {
    let a = quat_to_mat(q);
    [[[a[0], a[1]], [a[2], a[3]]], [[a[4], a[5]], [a[6], a[7]]]]
}

fn mdag(a: M) -> M {
    std::array::from_fn(|i| std::array::from_fn(|j| [a[j][i][0], -a[j][i][1]]))
}

fn mmul(a: M, b: M) -> M {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let x = cmul(a[i][0], b[0][j]);
            let y = cmul(a[i][1], b[1][j]);
            [x[0] + y[0], x[1] + y[1]]
        })
    })
}

fn madd(a: M, b: M) -> M {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| [a[i][j][0] + b[i][j][0], a[i][j][1] + b[i][j][1]])
    })
}

fn mscale(a: M, facteur: f64) -> M {
    a.map(|ligne| ligne.map(|z| [z[0] * facteur, z[1] * facteur]))
}

fn trace(a: M) -> f64 {
    a[0][0][0] + a[1][1][0]
}

fn determinant(a: M) -> C {
    let x = cmul(a[0][0], a[1][1]);
    let y = cmul(a[0][1], a[1][0]);
    [x[0] - y[0], x[1] - y[1]]
}

/// Voisin par coordonnées : ne réutilise ni nb_up/nb_dn, ni produit_chemin.
fn deplacer(geo: &Geo, s: usize, d: usize, sens: i32) -> usize {
    let mut x = geo.coords(s);
    x[d] = (x[d] as i32 + sens).rem_euclid(geo.l as i32) as usize;
    ((x[0] * geo.l + x[1]) * geo.l + x[2]) * geo.l + x[3]
}

fn chemin_matriciel(u: &[M], geo: &Geo, s: usize, chemin: &[(usize, i32)], m: usize) -> M {
    let mut pos = s;
    let mut p = IDENTITE;
    for &(d, sens) in chemin {
        for _ in 0..m {
            let lien = if sens > 0 {
                let a = u[d * geo.v + pos];
                pos = deplacer(geo, pos, d, 1);
                a
            } else {
                pos = deplacer(geo, pos, d, -1);
                mdag(u[d * geo.v + pos])
            };
            p = mmul(p, lien);
        }
    }
    p
}

fn proches(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "{a:.17e} ≠ {b:.17e}, écart {:.3e} > {tol:.3e}",
        (a - b).abs()
    );
}

#[test]
fn agrafes_contre_six_plaquettes_matricielles() {
    let c = champ4();
    let geo = Geo::new(c.l);
    let u = en_quaternions(c);
    let matrices: Vec<M> = u.iter().copied().map(matrice).collect();
    let mut alea = Alea(26102619004);
    for _ in 0..20 {
        let mu = alea.entier() as usize % 4;
        let s = alea.entier() as usize % geo.v;
        let w = w_agrafes(&u, &geo, mu, s);
        let lien = u[mu * geo.v + s];
        let trace_q = 2.0 * (lien[0] * w[0] - lien[1] * w[1] - lien[2] * w[2] - lien[3] * w[3]);
        let mut trace_m = 0.0;
        for nu in 0..4 {
            if nu != mu {
                for sens in [1, -1] {
                    trace_m += trace(chemin_matriciel(
                        &matrices,
                        &geo,
                        s,
                        &[(mu, 1), (nu, sens), (mu, -1), (nu, -sens)],
                        1,
                    ));
                }
            }
        }
        proches(trace_q, trace_m, 1e-13);
        proches(
            trace_q,
            trace(mmul(matrices[mu * geo.v + s], matrice(w))),
            1e-13,
        );
    }
    // Un lien central isolé ne doit jamais entrer dans son propre W.
    let mut u = vec![UN; 4 * geo.v];
    u[17] = [0.5, 0.5, 0.5, 0.5];
    assert_eq!(w_agrafes(&u, &geo, 0, 17), [6.0, 0.0, 0.0, 0.0]);
}

#[test]
fn trefle_un_et_charge_reference_au_bit() {
    for c in [champ4().clone(), thermalise(6, 2.5, 26102619005)] {
        let res = Reseau::nouveau(&c).unwrap();
        for s in 0..c.v() {
            let g = trefle_mxm(&res, s, 1);
            let reference = ref_cpu::champ_trefle(&res, s);
            for (a, b) in g.iter().flatten().zip(reference.iter().flatten()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            assert_eq!(
                densite_q(&g).to_bits(),
                ref_cpu::densite_trefle_site(&res, s).1.to_bits()
            );
        }
        let c_topo = charges(&c).unwrap();
        assert_eq!(
            c_topo.q_tre.to_bits(),
            ref_cpu::densites(&c).unwrap().q_top.to_bits()
        );
        assert_eq!(charge_trefle(&c).unwrap().to_bits(), c_topo.q_tre.to_bits());
        assert_eq!(
            charge_amelioree(&c).unwrap().to_bits(),
            c_topo.q_imp.to_bits()
        );
    }
}

#[test]
fn trefles_mxm_contre_matrices() {
    let geo = Geo::new(6);
    let mut alea = Alea(26102619006);
    let u: Vec<Q> = (0..4 * geo.v).map(|_| alea.haar()).collect();
    let c = en_champ(&u, geo.l);
    let res = Reseau::nouveau(&c).unwrap();
    let matrices: Vec<M> = u.into_iter().map(matrice).collect();
    let s = 1109; // Les deux tailles traversent des raccords périodiques.
    let plan = 4;
    let (mu, nu) = PLANS[plan];
    for m in [2, 3] {
        let mut somme = [[[0.0; 2]; 2]; 2];
        // Rotation du chemin +μ,+ν,−μ,−ν autour de ses quatre coins.
        let contour = [(mu, 1), (nu, 1), (mu, -1), (nu, -1)];
        for coin in 0..4 {
            let chemin: Vec<_> = (0..4).map(|k| contour[(coin + k) % 4]).collect();
            somme = madd(somme, chemin_matriciel(&matrices, &geo, s, &chemin, m));
        }
        let anti = mscale(madd(somme, mscale(mdag(somme), -1.0)), 1.0 / 8.0);
        let attendu = [anti[0][1][1], anti[0][1][0], anti[0][0][1]];
        let g = trefle_mxm(&res, s, m);
        for (a, b) in g[plan].into_iter().zip(attendu) {
            proches(a, b, 1e-13);
        }
    }
    let g = champ_ameliore(&res, s);
    let champs = [
        trefle_mxm(&res, s, 1),
        trefle_mxm(&res, s, 2),
        trefle_mxm(&res, s, 3),
    ];
    for p in 0..6 {
        for k in 0..3 {
            assert_eq!(
                g[p][k].to_bits(),
                (COEFS_AMELIORE[0] * champs[0][p][k] / 1.0
                    + COEFS_AMELIORE[1] * champs[1][p][k] / 4.0
                    + COEFS_AMELIORE[2] * champs[2][p][k] / 9.0)
                    .to_bits()
            );
        }
    }
}

#[test]
fn topo_inst_flux_jauge_et_reflexion() {
    let references = [
        (8, 1.993_582_728_089_921_6, 2.001_525_336_395_644),
        (12, 1.998_731_082_900_731_6, 2.000_060_618_064_416_5),
        (16, 1.999_598_437_023_194, 2.000_006_087_553_055),
    ];
    // Calculer et afficher les trois couples même si une gate échoue ensuite.
    let mesures: Vec<_> = references
        .iter()
        .map(|&(l, qt, qi)| {
            let c = champ_abelien(l);
            let q = charges(&c).unwrap();
            println!(
                "TOPO-INST L={l} : q_tre={:+.17}, q_imp={:+.17}",
                q.q_tre, q.q_imp
            );
            (c, q, qt, qi)
        })
        .collect();
    for (c, q, qt, qi) in mesures {
        assert!(q.q_imp > 0.0, "TOPO-INST L={} : signe opposé à celui attendu (+2), obtenu {:+.17} ; convention laissée intacte pour arbitrage du pilote", c.l, q.q_imp);
        proches(q.q_tre, qt, 1e-10);
        proches(q.q_imp, qi, 1e-10);
        assert!(q.q_imp - q.q_tre >= 4e-4);
        let jauge = charges(&transformer_jauge(&c, 26102619001)).unwrap();
        proches(jauge.q_tre, q.q_tre, 1e-10);
        proches(jauge.q_imp, q.q_imp, 1e-10);
        proches(jauge.plaq, q.plaq, 1e-13);
        let reflet = reflechir_axe3(&c);
        let qr = charges(&reflet).unwrap();
        proches(qr.q_tre, -q.q_tre, 1e-10);
        proches(qr.q_imp, -q.q_imp, 1e-10);
        assert_eq!(reflechir_axe3(&reflet), c);
    }
}

/// Cooling séquentiel matriciel : liens visités à rebours dans chaque classe.
fn cooling_matriciel(c: &ChampHote, balayages: usize) -> Vec<M> {
    let geo = Geo::new(c.l);
    let mut u: Vec<M> = en_quaternions(c).into_iter().map(matrice).collect();
    for _ in 0..balayages {
        for mu in 0..4 {
            for parite in 0..2 {
                for s in (0..geo.v).rev() {
                    let x = geo.coords(s);
                    if (x[0] + x[1] + x[2] + x[3]) % 2 != parite {
                        continue;
                    }
                    let depart = deplacer(&geo, s, mu, 1);
                    let mut w = [[[0.0; 2]; 2]; 2];
                    for nu in 0..4 {
                        if nu != mu {
                            for sens in [1, -1] {
                                w = madd(
                                    w,
                                    chemin_matriciel(
                                        &u,
                                        &geo,
                                        depart,
                                        &[(nu, sens), (mu, -1), (nu, -sens)],
                                        1,
                                    ),
                                );
                            }
                        }
                    }
                    let det = determinant(w);
                    proches(det[1], 0.0, 1e-13);
                    u[mu * geo.v + s] = if det[0] == 0.0 {
                        IDENTITE
                    } else {
                        mscale(mdag(w), 1.0 / det[0].sqrt())
                    };
                }
            }
        }
    }
    u
}

#[test]
fn cooling_action_unitarite_et_damier_matriciel() {
    let c = champ4();
    let initial = c.clone();
    let (serie, final_c) = cooling(c, 20, true).unwrap();
    assert_eq!(c, &initial);
    assert_eq!(serie.q_imp.len(), 21);
    assert_eq!(serie.q_tre.len(), 21);
    assert_eq!(serie.action.len(), 21);
    let avant = charges(c).unwrap();
    assert_eq!(serie.q_imp[0].to_bits(), avant.q_imp.to_bits());
    assert_eq!(serie.q_tre[0].to_bits(), avant.q_tre.to_bits());
    for paire in serie.action.windows(2) {
        assert!(
            paire[1] - paire[0] <= 1e-12 * paire[0].abs(),
            "action croissante : {paire:?}"
        );
    }
    assert!(final_c.defaut_unitarite() <= 1e-14);
    assert!(charges(&final_c).unwrap().plaq > avant.plaq);
    let (_, trois) = cooling(c, 3, true).unwrap();
    let reference = cooling_matriciel(c, 3);
    for (a, b) in en_quaternions(&trois)
        .into_iter()
        .map(matrice)
        .zip(reference)
    {
        for (x, y) in a
            .iter()
            .flatten()
            .flatten()
            .zip(b.iter().flatten().flatten())
        {
            proches(*x, *y, 1e-12);
        }
    }
    let (sans, copie) = cooling(c, 0, true).unwrap();
    assert_eq!(copie, *c);
    assert_eq!(sans.action, vec![serie.action[0]]);
    assert_eq!(maj_cooling([0.0; 4], true), UN);
}

#[test]
fn mutant_sans_dagger_degrade_la_plaquette() {
    let c = champ4();
    let (serie, final_c) = cooling(c, 20, false).unwrap();
    assert!(charges(&final_c).unwrap().plaq < charges(c).unwrap().plaq);
    assert!(serie.action[20] > serie.action[0]);
    assert!(serie.action.windows(2).any(|paire| paire[1] > paire[0]));
}

#[test]
fn cool_id_et_trace_matricielle() {
    let w = cool_id_matrices(26102619002, 1000);
    assert_eq!(w, cool_id_matrices(26102619002, 1000));
    assert!(cool_id_ecart(&w, true) <= 1e-13);
    assert!(cool_id_ecart(&w, false) >= 1e-2);
    for dagger in [true, false] {
        let mut ecart: f64 = 0.0;
        for &w in &w {
            let wm = matrice(w);
            let u = matrice(maj_cooling(w, dagger));
            let det = determinant(wm);
            proches(det[1], 0.0, 1e-13);
            ecart = ecart.max((trace(mmul(u, wm)) - 2.0 * det[0].sqrt()).abs());
        }
        proches(ecart, cool_id_ecart(&w, dagger), 1e-13);
    }
    assert_eq!(cool_id_ecart(&[], true), 0.0);
}

#[test]
fn philox_definition_determinisme_et_haar() {
    use kvu::heritage::su2_hb::{philox4x32_10, u01};
    let graine = 0x1234_5678_9abc_def0;
    let mut a = AleaPhilox::nouveau(graine);
    for i in 0..1000_u64 {
        let bloc = philox4x32_10(
            [i as u32, (i >> 32) as u32, 0, 0],
            graine as u32,
            (graine >> 32) as u32,
        );
        let u = a.uniforme();
        assert!(u > 0.0 && u < 1.0);
        assert_eq!(u.to_bits(), u01(bloc[0], bloc[1]).to_bits());
    }
    let mut a = AleaPhilox::nouveau(graine);
    let mut b = a.clone();
    for _ in 0..1000 {
        let attendu = (-2.0 * b.uniforme().ln()).sqrt() * (TAU * b.uniforme()).cos();
        assert_eq!(a.normale().to_bits(), attendu.to_bits());
        let q = a.haar_su2();
        proches(qnorm(q), 1.0, 5e-16);
        b.haar_su2();
    }
    // Vérification indépendante de l'ordre des tirages V puis r.
    let mut a = AleaPhilox::nouveau(graine);
    for w in cool_id_matrices(graine, 4) {
        let mut attendu = [0.0; 4];
        for _ in 0..6 {
            let v = a.haar_su2();
            let r = a.uniforme();
            for k in 0..4 {
                attendu[k] += r * v[k];
            }
        }
        assert_eq!(w, attendu);
    }
}

#[test]
fn mesure_complete_determinisme_et_reference() {
    let c = champ4();
    let copie = c.clone();
    let p = ParamsTopo {
        pas_max: 500,
        ..ParamsTopo::default()
    };
    let r = record4().clone();
    assert_eq!(r, mesurer_cpu(c, &p).unwrap());
    assert_eq!(c, &copie);
    assert!(r.ens.is_empty() && r.backend_gen.is_empty() && r.fichier.is_empty());
    assert_eq!((r.chaine, r.cfg, r.l), (0, 0, 4));
    assert_eq!(r.backend_mesure, "cpu");
    assert_eq!(r.plaq0.to_bits(), charges(c).unwrap().plaq.to_bits());
    assert_eq!(r.flot_a2.e_plaq.len(), 4);
    assert_eq!(r.flot_a2.q_imp.len(), 4);
    assert_eq!(r.flot_a2.q_tre.len(), 4);
    assert_eq!(r.cooling.q_imp.len(), 21);
    assert_eq!(r.cooling, cooling(c, 20, true).unwrap().0);
    let flotte = ref_cpu::flot_n_pas(c, 0.01, 300).unwrap();
    assert_eq!(
        r.flot_a2.q_tre[IDX_T3].to_bits(),
        charge_trefle(&flotte).unwrap().to_bits()
    );
    assert_eq!(
        r.flot_a2.q_imp[IDX_T3].to_bits(),
        charge_amelioree(&flotte).unwrap().to_bits()
    );
    assert_eq!(
        r.flot_a2.e_plaq[IDX_T3].to_bits(),
        ref_cpu::densites(&flotte).unwrap().e_plaq.to_bits()
    );
    assert_eq!(r.t0.is_none(), r.flot_t0.is_none());
}

#[test]
fn aller_retour_json_f64_exact() {
    let r = record4();
    let json = serde_json::to_string(r).unwrap();
    let relu: RecordTopo = serde_json::from_str(&json).unwrap();
    // Diagnostic ciblé : le parseur sans float_roundtrip peut décaler un f64 d'un bit.
    for (nom, avant, apres) in [
        ("flot_a2.e_plaq", &r.flot_a2.e_plaq, &relu.flot_a2.e_plaq),
        ("flot_a2.q_tre", &r.flot_a2.q_tre, &relu.flot_a2.q_tre),
        ("flot_a2.q_imp", &r.flot_a2.q_imp, &relu.flot_a2.q_imp),
        ("cooling.q_tre", &r.cooling.q_tre, &relu.cooling.q_tre),
        ("cooling.q_imp", &r.cooling.q_imp, &relu.cooling.q_imp),
        ("cooling.action", &r.cooling.action, &relu.cooling.action),
    ] {
        for (i, (a, b)) in avant.iter().zip(apres).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "JSON {nom}[{i}] : {a:.17e} devient {b:.17e} ; serde_json/float_roundtrip nécessaire");
        }
    }
    assert_eq!(r, &relu);
}

#[test]
fn t0_premier_croisement_grille_et_prolongation() {
    let c = thermalise(4, 0.5, 26102619007);
    // Cible basse : croisement garanti par le premier pas de ce champ rugueux.
    let p = ParamsTopo {
        cible_t2e: 0.0001,
        pas_max: 500,
        balayages: 0,
        ..ParamsTopo::default()
    };
    let r = mesurer_cpu(&c, &p).unwrap();
    let mut u = en_quaternions(&c);
    let geo = Geo::new(c.l);
    let mut temps = vec![0.0];
    let mut courbe = vec![0.0];
    for n in 1..=500 {
        ref_cpu::pas_rk3(&mut u, &geo, p.dt);
        let t = n as f64 * p.dt;
        temps.push(t);
        courbe.push(t * t * ref_cpu::densites(&en_champ(&u, geo.l)).unwrap().e_plaq);
        if courbe[n] >= p.cible_t2e {
            break;
        }
    }
    let t0 = echelles(&temps, &courbe, p.cible_t2e, 0.3).0.unwrap();
    assert_eq!(r.t0.unwrap().to_bits(), t0.to_bits());
    let f = r.flot_t0.unwrap();
    assert_eq!(f.t_eff.len(), 4);
    for (i, k) in MULT_T0.into_iter().enumerate() {
        let n = (k * t0 / p.dt - 1e-9).ceil() as usize;
        assert_eq!(f.t_eff[i].to_bits(), (n as f64 * p.dt).to_bits());
        assert!(f.t_eff[i] + 1e-11 >= k * t0);
        let attendu = charges(&ref_cpu::flot_n_pas(&c, p.dt, n).unwrap()).unwrap();
        assert_eq!(f.q_tre[i].to_bits(), attendu.q_tre.to_bits());
        assert_eq!(f.q_imp[i].to_bits(), attendu.q_imp.to_bits());
    }
    // Le flux abélien est stationnaire : t₀ ≃ 2,22, donc 5t₀ dépasse pas_max·dt.
    let c = champ_abelien(4);
    let p = ParamsTopo {
        cible_t2e: 3.0,
        pas_max: 500,
        balayages: 0,
        ..ParamsTopo::default()
    };
    let r = mesurer_cpu(&c, &p).unwrap();
    let t0 = r.t0.unwrap();
    let f = r.flot_t0.unwrap();
    assert!(f.t_eff[3] > p.pas_max as f64 * p.dt);
    for (i, k) in MULT_T0.into_iter().enumerate() {
        let n = (k * t0 / p.dt - 1e-9).ceil() as usize;
        assert_eq!(f.t_eff[i].to_bits(), (n as f64 * p.dt).to_bits());
    }
    let n = (f.t_eff[3] / p.dt).round() as usize;
    let q = charges(&ref_cpu::flot_n_pas(&c, p.dt, n).unwrap()).unwrap();
    assert_eq!(f.q_imp[3].to_bits(), q.q_imp.to_bits());
}

#[test]
fn t0_absent_sur_champ_froid() {
    let c = ChampHote::froid(4).unwrap();
    let p = ParamsTopo {
        pas_max: 500,
        balayages: 0,
        ..ParamsTopo::default()
    };
    let r = mesurer_cpu(&c, &p).unwrap();
    assert_eq!(r.t0, None);
    assert_eq!(r.flot_t0, None);
    assert_eq!(r.plaq0, 1.0);
    assert_eq!(r.flot_a2.e_plaq, vec![0.0; 4]);
    assert_eq!(r.flot_a2.q_tre, vec![0.0; 4]);
    assert_eq!(r.flot_a2.q_imp, vec![0.0; 4]);
    assert_eq!(
        r,
        serde_json::from_str::<RecordTopo>(&serde_json::to_string(&r).unwrap()).unwrap()
    );
}

#[test]
fn parametres_invalides() {
    let c = champ4();
    for p in [
        ParamsTopo {
            pas_max: 499,
            ..ParamsTopo::default()
        },
        ParamsTopo {
            dt: 0.0,
            ..ParamsTopo::default()
        },
        ParamsTopo {
            dt: -0.01,
            ..ParamsTopo::default()
        },
        ParamsTopo {
            dt: f64::NAN,
            ..ParamsTopo::default()
        },
        ParamsTopo {
            dt: f64::INFINITY,
            ..ParamsTopo::default()
        },
        ParamsTopo {
            cible_t2e: f64::NAN,
            ..ParamsTopo::default()
        },
        ParamsTopo {
            cible_t2e: 0.0,
            ..ParamsTopo::default()
        },
    ] {
        assert!(matches!(mesurer_cpu(c, &p), Err(Erreur::Config(_))));
    }
    let invalide = ChampHote { l: 4, q: vec![] };
    assert!(charges(&invalide).is_err());
    assert!(cooling(&invalide, 0, true).is_err());
    assert!(mesurer_cpu(&invalide, &ParamsTopo::default()).is_err());
    let impair = en_champ(&vec![UN; 4 * 3_usize.pow(4)], 3);
    assert!(matches!(cooling(&impair, 1, true), Err(Erreur::Config(_))));
}

#[test]
fn mutant_charge_demi_entier() {
    for q in [-3.09_f64, -2.0, -0.01, 0.0, 1.09, 2.0, 5.01] {
        let mutant = decaler_charge(q);
        assert_eq!(mutant.to_bits(), (q + 0.5).to_bits());
        assert!((q - q.round()).abs() < 0.1);
        assert!((mutant - mutant.round()).abs() >= 0.4);
    }
}

/// Avenant v4.3.1 §B : flot partiel (3 valeurs, pas de t₀) identique au bit au flot complet sur t/a² = 1, 2, 3 ; cooling identique.
#[test]
fn flot_partiel_identique_au_complet_sur_les_trois_premiers_temps() {
    let c = thermalise(4, 2.5, 26102619011);
    let complet = mesurer_cpu(&c, &ParamsTopo::default()).unwrap();
    let partiel = mesurer_cpu(&c, &ParamsTopo { flot_complet: false, ..ParamsTopo::default() }).unwrap();
    assert_eq!(partiel.flot_a2.q_imp.len(), 3);
    assert_eq!(partiel.flot_a2.q_tre.len(), 3);
    assert_eq!(partiel.flot_a2.e_plaq.len(), 3);
    assert!(partiel.t0.is_none() && partiel.flot_t0.is_none());
    for k in 0..3 {
        assert_eq!(partiel.flot_a2.q_imp[k].to_bits(), complet.flot_a2.q_imp[k].to_bits());
        assert_eq!(partiel.flot_a2.q_tre[k].to_bits(), complet.flot_a2.q_tre[k].to_bits());
        assert_eq!(partiel.flot_a2.e_plaq[k].to_bits(), complet.flot_a2.e_plaq[k].to_bits());
    }
    assert_eq!(partiel.cooling, complet.cooling);
    assert_eq!(partiel.plaq0.to_bits(), complet.plaq0.to_bits());
}
