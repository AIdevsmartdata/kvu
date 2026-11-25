//! `thermo mesure --lien-integre` : estimateur à lien intégré (multi-hit PPR 1983). Avec carte (`--ignored`) : carte contre référence hôte
//! sur configuration figée (aléatoire, puis thermalisée), F64 et F32 ; export pour la vérification numpy (LI_EXPORT=dossier).
use kvu::commun::{BackendKind, Depart, Graine, Precision};
use kvu::gpu::{self, Appareil};
use kvu::heritage::su2_hb::{qdag, qmul, Q};
use kvu::su2::torsion::Torsion;
use kvu::su2::une_boite::{ParamBoite, UneBoite};
use kvu::su2::ParamChaine;

fn appareil() -> Appareil {
    let b = if cfg!(feature = "cuda") { BackendKind::Cuda } else { BackendKind::Hip };
    gpu::ouvrir(b, 0).unwrap_or_else(|e| panic!("FAIL : backend {} indisponible ({e})", b.nom()))
}
fn pb(d: [usize; 4], beta: f64, pr: Precision) -> ParamBoite {
    ParamBoite {
        cotes: d,
        torsion: Torsion::analyser("aucune").unwrap(),
        poids: 1.0,
        nor: 2,
        chaines: vec![ParamChaine { beta, graine: Graine(5), depart: Depart::Chaud }],
        precision: pr,
        impairs: false,
        aniso: None,
        couture: None,
    }
}

/// I2(k)/I1(k) par séries entières (méthode INDÉPENDANTE de la fraction continue du noyau).
fn ratio_serie(k: f64) -> f64 {
    if k <= 0.0 {
        return 0.0;
    }
    let (mut i1, mut i2) = (0.0f64, 0.0f64);
    let h = 0.5 * k;
    let mut t1 = h; // (k/2)^1 / (0! 1!)
    let mut t2 = h * h / 2.0; // (k/2)^2 / (0! 2!)
    for m in 0..2000 {
        i1 += t1;
        i2 += t2;
        let mf = m as f64;
        t1 *= h * h / ((mf + 1.0) * (mf + 2.0));
        t2 *= h * h / ((mf + 1.0) * (mf + 3.0));
        if t1 < 1e-30 * i1 && t2 < 1e-30 * i2 {
            break;
        }
    }
    i2 / i1
}

/// Référence hôte : (S_tot, S_t, S_s) bruts et à lien intégré, temps = direction 0, poids de plan (1 espace ; wt temps), β de chaîne.
fn hote(d: [usize; 4], q: &[f64], beta: f64, wt: f64) -> ([f64; 3], [f64; 3]) {
    let v: usize = d.iter().product();
    let co = |s: usize| {
        let mut x = [0; 4];
        let mut t = s;
        for k in (0..4).rev() {
            x[k] = t % d[k];
            t /= d[k];
        }
        x
    };
    let ix = |x: [usize; 4]| ((x[0] * d[1] + x[1]) * d[2] + x[2]) * d[3] + x[3];
    let hu = |x: [usize; 4], m: usize| {
        let mut y = x;
        y[m] = (y[m] + 1) % d[m];
        ix(y)
    };
    let bs = |x: [usize; 4], m: usize| {
        let mut y = x;
        y[m] = (y[m] + d[m] - 1) % d[m];
        ix(y)
    };
    let l = |m: usize, s: usize| -> Q {
        let i = (m * v + s) * 4;
        [q[i], q[i + 1], q[i + 2], q[i + 3]]
    };
    let add = |a: Q, b: Q| -> Q { [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]] };
    let sc = |a: Q, z: f64| -> Q { [a[0] * z, a[1] * z, a[2] * z, a[3] * z] };
    let dot = |a: Q, b: Q| a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let (mut brut, mut li) = ([0.0; 3], [0.0; 3]);
    for s in 0..v {
        let x = co(s);
        for mu in 0..4 {
            for nu in (mu + 1)..4 {
                let a = qmul(l(mu, s), l(nu, hu(x, mu)));
                let b = qmul(qdag(l(mu, hu(x, nu))), qdag(l(nu, s)));
                let val = qmul(a, b)[0];
                brut[0] += val;
                if mu == 0 { brut[1] += val } else { brut[2] += val }
            }
        }
        for mu in 0..4 {
            let (mut a, mut bt, mut bsp) = ([0.0; 4], [0.0; 4], [0.0; 4]);
            let smu = hu(x, mu);
            for nu in 0..4 {
                if nu == mu {
                    continue;
                }
                let snu = hu(x, nu);
                let sdn = bs(x, nu);
                let xd = co(sdn);
                let smu_dn = bs(co(smu), nu);
                let _ = xd;
                let t = qmul(qmul(l(nu, smu), qdag(l(mu, snu))), qdag(l(nu, s)));
                let b = qmul(qmul(qdag(l(nu, smu_dn)), qdag(l(mu, sdn))), l(nu, sdn));
                let w = if mu == 0 || nu == 0 { wt } else { 1.0 };
                a = add(a, sc(add(t, b), w));
                if mu == 0 || nu == 0 { bt = add(bt, add(t, b)) } else { bsp = add(bsp, add(t, b)) }
            }
            let an = dot(a, a).sqrt();
            if an > 1e-300 {
                let c = ratio_serie(beta * an) / an;
                let (dt, ds) = (dot(a, bt), dot(a, bsp));
                if mu == 0 {
                    li[0] += 0.5 * c * dt; // Y0 : liens temporels
                } else {
                    li[1] += 0.5 * c * dt; // Ys : liens spatiaux, S_t
                    li[2] += 0.25 * c * ds; // Zs : liens spatiaux, S_s
                }
            }
        }
    }
    (brut, li)
}

fn champ_aleatoire(v: usize, graine: u64) -> Vec<f64> {
    let mut z = graine;
    let mut nxt = || {
        z = z.wrapping_add(0x9E3779B97F4A7C15);
        let mut t = z;
        t = (t ^ (t >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        t = (t ^ (t >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((t ^ (t >> 31)) >> 11) as f64 / 9007199254740992.0 - 0.5
    };
    let mut q = vec![];
    for _ in 0..4 * v {
        let w = [nxt(), nxt(), nxt(), nxt()];
        let n = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        q.extend(w.iter().map(|x| x / n));
    }
    q
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / (1.0 + b.abs())
}

fn cas(pr: Precision, tol: f64) {
    let app = appareil();
    let d = [4, 6, 4, 4];
    let v: usize = d.iter().product();
    for (beta, wt, st) in [(2.0, 1.0, false), (2.0, 0.3, true), (1.2, 1.0, true), (2.073, 0.0, true), (0.5, 1.0, false)] {
        for thermalise in [false, true] {
            let p = pb(d, beta, pr);
            let mut m = UneBoite::nouveau_li(&app, &p, if st { Some((1.0, wt)) } else { None }).unwrap();
            m.init().unwrap();
            if thermalise {
                m.serie_li(15, 2, 1).unwrap();
            } else {
                m.televerser(0, &champ_aleatoire(v, 42)).unwrap();
            }
            m.allouer_historique(1).unwrap();
            m.mesurer(0).unwrap();
            let h = m.historique().unwrap();
            let hl = m.historique_li().unwrap();
            let q = m.telecharger(0).unwrap();
            let (brut, li) = hote(d, &q, beta, wt);
            assert!(rel(h[0], brut[0]) < tol, "S_tot brut carte {} hote {}", h[0], brut[0]);
            for k in 0..3 {
                assert!(rel(hl[k], li[k]) < tol, "pr {pr:?} beta {beta} wt {wt} therm {thermalise} composante {k} : carte {} hote {}", hl[k], li[k]);
            }
            if wt == 0.0 {
                assert_eq!(hl[0], 0.0, "Y0 exactement nul a beta_t = 0");
            }
            if let Ok(dir) = std::env::var("LI_EXPORT") {
                let nom = format!("{dir}/cfg_{:?}_{beta}_{wt}_{thermalise}.json", pr);
                let j = serde_json::json!({"d": d, "beta": beta, "wt": wt, "q": q, "carte_li": hl, "carte_brut_tot": h[0], "hote_li": li});
                std::fs::write(nom, j.to_string()).unwrap();
            }
            println!("PASS beta {beta} wt {wt} therm {thermalise} : li carte {:?} hote {:?}", hl, li);
        }
    }
}

#[test]
#[ignore = "GPU"]
fn li_carte_egale_hote_f64() {
    cas(Precision::F64, 1e-11);
}

#[test]
#[ignore = "GPU"]
fn li_carte_egale_hote_f32() {
    cas(Precision::F32, 1e-6);
}

/// Étalon de Codex (tour 5) : tous les liens à l'identité : Y0 − Ys = 3 L³ T [R(6 β_t) − R(4 β_s + 2 β_t)] (Y0 = 3V R(6βt), Ys = 3V R(4βs+2βt)) ;
/// à β_t = 0, Y0 = 0 sur la configuration (identité ET aléatoire : test cas()).
fn etalon(pr: Precision, tol: f64) {
    let app = appareil();
    let d = [4, 6, 4, 4];
    let v: usize = d.iter().product();
    for (bs, bt) in [(2.073, 0.19), (2.073, 0.6), (1.0, 1.0), (2.5, 1.8)] {
        let mut m = UneBoite::nouveau_li(&app, &pb(d, bs, pr), Some((1.0, bt / bs))).unwrap();
        m.init().unwrap();
        let mut q = vec![0.0; 16 * v];
        for i in 0..4 * v {
            q[4 * i] = 1.0;
        }
        m.televerser(0, &q).unwrap();
        m.allouer_historique(1).unwrap();
        m.mesurer(0).unwrap();
        let hl = m.historique_li().unwrap();
        let attendu = 3.0 * v as f64 * (ratio_serie(6.0 * bt) - ratio_serie(4.0 * bs + 2.0 * bt));
        assert!(((hl[0] - hl[1]) - attendu).abs() < tol * (1.0 + attendu.abs()), "{bs} {bt} : Y0 - Ys = {} attendu {attendu}", hl[0] - hl[1]);
        println!("PASS etalon identite ({bs}, {bt}) : Y0 - Ys = {} attendu {attendu}", hl[0] - hl[1]);
    }
}
#[test]
#[ignore = "GPU"]
fn li_etalon_identite_f64() {
    etalon(Precision::F64, 1e-11);
}
#[test]
#[ignore = "GPU"]
fn li_etalon_identite_f32() {
    etalon(Precision::F32, 1e-5);
}

/// Sans carte : le rapport I2/I1 par séries (référence) sur des valeurs connues.
#[test]
fn ratio_serie_valeurs() {
    // petit k : k/4 ; k = 1 : I2(1)/I1(1) = 0.135747669767038 / 0.565159103992485
    assert!((ratio_serie(1e-6) - 0.25e-6).abs() < 1e-18);
    assert!((ratio_serie(1.0) - 0.135747669767038 / 0.565159103992485).abs() < 1e-14);
}
