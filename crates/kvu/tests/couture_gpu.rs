//! Couture décalée (tore oblique) SUR CARTE : `su2 thermo mesure --decalage` (M7-GPU). Branche `geometrie/coutures-gpu-1122`.
//! Sans carte : tables et refus. Avec carte (`--ignored`, sous le verrou de la carte) : décalage nul ou période entière = octets du
//! chemin existant ; carte = hôte sur champ imposé (action, invariance de jauge) ; carte = processeur (`su2 gen --backend cpu
//! --decalage`, la référence) sur des chaînes de bain thermique + surrelaxation ; F32 contre F64.

#![allow(clippy::needless_range_loop)]

use kvu::commun::{BackendKind, Depart, Graine, Precision};
use kvu::geom::GeoA;
use kvu::gpu::{self, Appareil};
use kvu::heritage::su2_hb::{qdag, qmul, Q};
use kvu::su2::torsion::Torsion;
use kvu::su2::une_boite::{geo_couture, tableau_geo_classes, tables_boite, ParamBoite, UneBoite};
use kvu::su2::ParamChaine;
use std::process::Command;

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
/// S_tot hôte = Σ_{μ<ν} ½ReTr U_p, voisins du tore OBLIQUE (`GeoA::decaler` suit la couture).
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

fn pb(cotes: [usize; 4], couture: Option<(usize, [i64; 4])>, chaines: Vec<ParamChaine>, prec: Precision) -> ParamBoite {
    ParamBoite {
        cotes,
        torsion: Torsion::analyser("aucune").unwrap(),
        poids: 1.0,
        nor: 1,
        chaines,
        precision: prec,
        impairs: true,
        aniso: None,
        couture,
    }
}
fn ch(beta: f64, g: u64, d: Depart) -> ParamChaine {
    ParamChaine { beta, graine: Graine(g), depart: d }
}

/// (côtés, axe, décalage) : pairs et impairs, axes 0 à 3, décalages positifs et négatifs.
const CAS: [([usize; 4], usize, [i64; 4]); 7] = [
    ([3, 12, 8, 8], 0, [0, -3, 0, 0]),   // M2 A
    ([6, 12, 8, 8], 0, [0, -6, 0, 0]),   // M2 B
    ([4, 4, 4, 4], 0, [0, 1, 0, 1]),
    ([4, 6, 4, 4], 1, [1, 0, 3, 0]),
    ([3, 4, 5, 4], 2, [1, 2, 0, 3]),
    ([4, 4, 4, 6], 3, [2, 1, 3, 0]),
    ([5, 4, 3, 4], 0, [0, 2, 1, 3]),
];

// ---------------------------------------------------------------- sans carte
#[test]
fn tables_couture_et_refus() {
    for (d, axe, vec) in CAS {
        let p = pb(d, Some((axe, vec)), vec![ch(2.0, 1, Depart::Chaud)], Precision::F64);
        let (t, _, _) = tables_boite(&p).unwrap_or_else(|e| panic!("{d:?} axe {axe} {vec:?} : {e}"));
        let g = geo_couture(&p).unwrap().expect("couture non nulle");
        g.verifier_classes(&t.classes).unwrap();
        assert!(t.classes.len() <= 3, "{d:?}");
        let geo = tableau_geo_classes(&t);
        assert!(geo[9] >= 2, "liste de classes explicite avec couture {d:?}");
    }
    // décalage nul ou période entière : réseau ordinaire, tampon geo INCHANGÉ
    for (d, axe, vec) in [([3, 12, 8, 8], 0usize, [0i64, 0, 0, 0]), ([3, 12, 8, 8], 0, [0, 12, 0, 0]), ([4, 4, 4, 4], 1, [4, 0, -8, 0])] {
        let p = pb(d, Some((axe, vec)), vec![ch(2.0, 1, Depart::Chaud)], Precision::F64);
        assert!(geo_couture(&p).unwrap().is_none());
        let q = pb(d, None, vec![ch(2.0, 1, Depart::Chaud)], Precision::F64);
        assert_eq!(tableau_geo_classes(&tables_boite(&p).unwrap().0), tableau_geo_classes(&tables_boite(&q).unwrap().0));
    }
    // refus : torsion, composante selon l'axe non nulle
    let mut p = pb([4, 4, 4, 4], Some((0, [0, 1, 0, 1])), vec![ch(2.0, 1, Depart::Chaud)], Precision::F64);
    p.torsion = Torsion::analyser("01").unwrap();
    assert!(tables_boite(&p).is_err());
    let p = pb([4, 4, 4, 4], Some((0, [1, 1, 0, 1])), vec![ch(2.0, 1, Depart::Chaud)], Precision::F64);
    assert!(tables_boite(&p).is_err());
    let mut p = pb([4, 4, 4, 4], Some((0, [0, 1, 0, 1])), vec![ch(2.0, 1, Depart::Chaud)], Precision::F64);
    p.aniso = Some((1.5, 3));
    assert!(tables_boite(&p).is_err());
}

// ---------------------------------------------------------------- carte
fn appareil() -> Appareil {
    let b = if cfg!(feature = "cuda") { BackendKind::Cuda } else { BackendKind::Hip };
    gpu::ouvrir(b, 0).unwrap_or_else(|e| panic!("FAIL : backend {} indisponible ({e})", b.nom()))
}
fn mesure_carte(m: &mut UneBoite, c: usize) -> f64 {
    m.allouer_historique(1).unwrap();
    m.mesurer(0).unwrap();
    m.historique().unwrap()[2 * c]
}

/// Décalage nul ou période entière : champ après 8 pas IDENTIQUE au bit à celui du chemin existant (F64 et F32).
#[test]
#[ignore = "GPU"]
fn g1_decalage_nul_identique_au_bit() {
    let app = appareil();
    for prec in [Precision::F64, Precision::F32] {
        for (d, axe, vec) in [([3, 12, 8, 8], 0usize, [0i64, 0, 0, 0]), ([3, 12, 8, 8], 0, [0, -12, 0, 0]), ([4, 4, 4, 4], 2, [4, 0, 0, -4]), ([3, 4, 5, 4], 1, [0, 0, 5, 0])] {
            let cs = vec![ch(2.1, 3, Depart::Chaud), ch(2.5, 4, Depart::Froid)];
            let mut a = UneBoite::nouveau(&app, &pb(d, None, cs.clone(), prec)).unwrap();
            let mut b = UneBoite::nouveau(&app, &pb(d, Some((axe, vec)), cs, prec)).unwrap();
            assert_eq!(a.empreinte_source, b.empreinte_source, "source ordinaire");
            a.init().unwrap();
            b.init().unwrap();
            for _ in 0..8 {
                a.pas().unwrap();
                b.pas().unwrap();
            }
            for c in 0..2 {
                let (qa, qb) = (a.telecharger(c).unwrap(), b.telecharger(c).unwrap());
                assert!(qa.iter().zip(&qb).all(|(x, y)| x.to_bits() == y.to_bits()), "{d:?} {prec:?} chaîne {c}");
            }
        }
    }
    println!("PASS G1");
}

/// Carte = hôte sur champ imposé (F64) : action sur le tore oblique, puis invariance de jauge (g(x) U g(x+μ)†, voisins obliques).
#[test]
#[ignore = "GPU"]
fn g2_action_hote_et_jauge_sur_tore_oblique() {
    let app = appareil();
    for (d, axe, vec) in CAS {
        let p = pb(d, Some((axe, vec)), vec![ch(2.3, 1, Depart::Chaud)], Precision::F64);
        let g = geo_couture(&p).unwrap().unwrap();
        let q = champ_haar(g.v, 17);
        let mut m = UneBoite::nouveau(&app, &p).unwrap();
        m.init().unwrap();
        m.televerser(0, &q).unwrap();
        let carte = mesure_carte(&mut m, 0);
        let hote = s_tot(&g, &q);
        assert!((carte - hote).abs() < 1e-11 * hote.abs().max(1.0) * (g.v as f64).sqrt(), "{d:?} {axe} {vec:?} : carte {carte} hôte {hote}");
        // le tore ORDINAIRE donne une autre valeur : le test voit la couture
        let go = GeoA::nouvelle(d).unwrap();
        assert!((s_tot(&go, &q) - hote).abs() > 1e-6 * hote.abs(), "couture invisible {d:?}");
        let mut r = Rng(77);
        let gs: Vec<Q> = (0..g.v).map(|_| r.haar()).collect();
        let mut q2 = q.clone();
        for s in 0..g.v {
            for mu in 0..4 {
                pose(g.v, &mut q2, mu, s, qmul(qmul(gs[s], lien(g.v, &q, mu, s)), qdag(gs[g.decaler(s, mu, 1)])));
            }
        }
        m.televerser(0, &q2).unwrap();
        let carte2 = mesure_carte(&mut m, 0);
        assert!((carte2 - carte).abs() < 1e-10 * carte.abs(), "jauge {d:?} : {carte2} contre {carte}");
        // un pas (bain + surrelaxation) garde les liens unitaires
        m.televerser(0, &q).unwrap();
        m.pas().unwrap();
        let q3 = m.telecharger(0).unwrap();
        for s in 0..g.v {
            for mu in 0..4 {
                let n: f64 = lien(g.v, &q3, mu, s).iter().map(|x| x * x).sum();
                assert!((n - 1.0).abs() < 1e-12);
            }
        }
        println!("PASS G2 {d:?} axe {axe} {vec:?}");
    }
}

fn kvu(args: &[String]) -> (i32, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_kvu")).args(args).env("KEVINOTRON_VERROU", "ignorer").output().unwrap();
    let mut s = String::from_utf8_lossy(&o.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&o.stderr));
    (o.status.code().unwrap_or(-1), s)
}
fn sv(a: &[&str]) -> Vec<String> {
    a.iter().map(|s| s.to_string()).collect()
}
fn liste(d: &[usize; 4]) -> String {
    d.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
}
fn liste_i(d: &[i64; 4]) -> String {
    d.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
}

/// Carte = processeur (`su2 gen --backend cpu --decalage`, référence) : plaquette moyenne de chaque pas d'une chaîne chaude
/// (bain + 1 surrelaxation), 6 pas ; même graine, même clé Philox par lien : accord à 1e-9 relatif.
#[test]
#[ignore = "GPU"]
fn g3_carte_egale_processeur() {
    let b = if cfg!(feature = "cuda") { "cuda" } else { "hip" };
    let base = std::env::temp_dir().join(format!("kvu_couture_g3_{}", std::process::id()));
    let mut pire = 0.0f64;
    for (i, (d, axe, vec)) in CAS.iter().enumerate() {
        let v: usize = d.iter().product();
        // Le processeur colorie au plus glouton (jusqu'à 9 classes) ; la carte à 3 classes au plus. Si les classes diffèrent, l'ordre
        // du balayage diffère : les chaînes ne sont plus identiques au bit (G5 les compare statistiquement).
        let g = GeoA::avec_couture(*d, *axe, *vec).unwrap();
        let p = pb(*d, Some((*axe, *vec)), vec![ch(2.0, 1, Depart::Chaud)], Precision::F64);
        let (t, _, _) = tables_boite(&p).unwrap();
        if t.classes != g.couleurs() {
            println!("SKIP G3 {d:?} axe {axe} {vec:?} : classes carte ({}) différentes du processeur ({}) : voir G5", t.classes.len(), g.couleurs().len());
            continue;
        }
        let (c, o) = (base.join(format!("cpu{i}")), base.join(format!("gpu{i}")));
        let mut a = sv(&["su2", "gen", "--cotes", &liste(d), "--decalage", &liste_i(vec), "--decalage-axe", &axe.to_string(), "--backend", "cpu", "--beta", "2.073", "--graine", "5", "--therm", "0", "--nmeas", "6", "--skip", "1", "--nor", "1", "--outdir", c.to_str().unwrap()]);
        a.push("--start".into());
        a.push("hot".into());
        let (code, t) = kvu(&a);
        assert_eq!(code, 0, "{t}");
        let a = sv(&["su2", "thermo", "mesure", "--cotes", &liste(d), "--decalage", &liste_i(vec), "--decalage-axe", &axe.to_string(), "--impairs", "oui", "--betas", "2.073", "--repliques", "2", "--therm", "0", "--nmeas", "6", "--skip", "1", "--nor", "1", "--graine", "5", "--depart", "chaud", "--backend", b, "--precision", "f64", "--series", "--sortie", o.to_str().unwrap()]);
        let (code, t) = kvu(&a);
        assert_eq!(code, 0, "{t}");
        let cpu: Vec<f64> = std::fs::read_to_string(c.join("measurements.csv"))
            .unwrap()
            .lines()
            .filter(|l| l.contains(",sample,"))
            .map(|l| l.split(',').nth(3).unwrap().parse().unwrap())
            .collect();
        let raw = std::fs::read(o.join("series/bras0_g000_tot.bin")).unwrap();
        let gpu: Vec<f64> = raw.chunks_exact(8).map(|x| f64::from_le_bytes(x.try_into().unwrap())).collect();
        assert_eq!(cpu.len(), 6);
        for k in 0..6 {
            let pg = gpu[2 * k] / (6.0 * v as f64); // réplique 0 (graine 5) : mêmes entrées que gen
            let e = (pg - cpu[k]).abs() / cpu[k].abs();
            pire = pire.max(e);
            assert!(e < 1e-9, "{d:?} {axe} {vec:?} pas {k} : carte {pg} processeur {} (écart {e:e})", cpu[k]);
        }
        println!("PASS G3 {d:?} axe {axe} {vec:?}");
    }
    println!("pire écart relatif carte/processeur : {pire:e}");
    let _ = std::fs::remove_dir_all(&base);
}

/// F32 contre F64 : plaquette moyenne (24 répliques x 200 mesures) des deux géométries de M2, β = 2,073 : écart < 4 écarts-types.
#[test]
#[ignore = "GPU"]
fn g4_f32_contre_f64() {
    let b = if cfg!(feature = "cuda") { "cuda" } else { "hip" };
    let base = std::env::temp_dir().join(format!("kvu_couture_g4_{}", std::process::id()));
    let mut res: Vec<(String, f64, f64)> = vec![];
    for prec in ["f64", "f32"] {
        let o = base.join(prec);
        let a = sv(&["su2", "thermo", "mesure", "--cotes", "3,8,6,6", "--decalage", "0,-3,0,0", "--impairs", "oui", "--betas", "2.073", "--repliques", "24", "--therm", "100", "--nmeas", "200", "--skip", "1", "--nor", "1", "--graine", "40", "--depart", "chaud", "--backend", b, "--precision", prec, "--sortie", o.to_str().unwrap()]);
        let (code, t) = kvu(&a);
        assert_eq!(code, 0, "{t}");
        let txt = std::fs::read_to_string(o.join("points/bras0_g000.csv")).unwrap();
        let x: Vec<f64> = txt.lines().skip(1).map(|l| l.split(',').nth(6).unwrap().parse::<f64>().unwrap() / (6.0 * 3.0 * 8.0 * 36.0)).collect();
        let m = x.iter().sum::<f64>() / x.len() as f64;
        let s2 = x.iter().map(|y| (y - m).powi(2)).sum::<f64>() / (x.len() - 1) as f64 / x.len() as f64;
        res.push((prec.to_string(), m, s2.sqrt()));
    }
    println!("{res:?}");
    let (e, s) = ((res[0].1 - res[1].1).abs(), (res[0].2.powi(2) + res[1].2.powi(2)).sqrt());
    assert!(e < 4.0 * s, "F64 {:?} contre F32 {:?} : écart {e} pour σ = {s}", res[0], res[1]);
    let _ = std::fs::remove_dir_all(&base);
}

/// G5 : cas où les classes diffèrent (carte à 3 classes, processeur glouton à 4) : plaquette moyenne carte contre processeur
/// (16 chaînes de 300 mesures chacune, 100 pas jetés), écart < 4 écarts-types combinés.
#[test]
#[ignore = "GPU"]
fn g5_classes_differentes_statistique() {
    let b = if cfg!(feature = "cuda") { "cuda" } else { "hip" };
    let base = std::env::temp_dir().join(format!("kvu_couture_g5_{}", std::process::id()));
    let (d, axe, vec) = ([3usize, 4, 5, 4], 2usize, [1i64, 2, 0, 3]);
    let v: usize = d.iter().product();
    let moy = |x: &[f64]| -> (f64, f64) {
        let m = x.iter().sum::<f64>() / x.len() as f64;
        (m, ((x.iter().map(|y| (y - m).powi(2)).sum::<f64>()) / (x.len() - 1) as f64 / x.len() as f64).sqrt())
    };
    let mut cpu = vec![];
    for r in 0..16u64 {
        let c = base.join(format!("cpu{r}"));
        let a = sv(&["su2", "gen", "--cotes", &liste(&d), "--decalage", &liste_i(&vec), "--decalage-axe", &axe.to_string(), "--backend", "cpu", "--beta", "2.073", "--graine", &(300 + r).to_string(), "--therm", "100", "--nmeas", "300", "--skip", "1", "--nor", "1", "--outdir", c.to_str().unwrap()]);
        let (code, t) = kvu(&a);
        assert_eq!(code, 0, "{t}");
        let x: Vec<f64> = std::fs::read_to_string(c.join("measurements.csv")).unwrap().lines().filter(|l| l.contains(",sample,")).map(|l| l.split(',').nth(3).unwrap().parse().unwrap()).collect();
        cpu.push(x.iter().sum::<f64>() / x.len() as f64);
    }
    let o = base.join("gpu");
    let a = sv(&["su2", "thermo", "mesure", "--cotes", &liste(&d), "--decalage", &liste_i(&vec), "--decalage-axe", &axe.to_string(), "--impairs", "oui", "--betas", "2.073", "--repliques", "16", "--therm", "100", "--nmeas", "300", "--skip", "1", "--nor", "1", "--graine", "500", "--depart", "chaud", "--backend", b, "--precision", "f64", "--sortie", o.to_str().unwrap()]);
    let (code, t) = kvu(&a);
    assert_eq!(code, 0, "{t}");
    let gpu: Vec<f64> = std::fs::read_to_string(o.join("points/bras0_g000.csv")).unwrap().lines().skip(1).map(|l| l.split(',').nth(6).unwrap().parse::<f64>().unwrap() / (6.0 * v as f64)).collect();
    let ((mc, sc), (mg, sg)) = (moy(&cpu), moy(&gpu));
    println!("G5 : processeur {mc:.5} +- {sc:.5} ; carte {mg:.5} +- {sg:.5}");
    assert!((mc - mg).abs() < 4.0 * (sc * sc + sg * sg).sqrt(), "carte {mg} contre processeur {mc}");
    let _ = std::fs::remove_dir_all(&base);
}
