//! `thermo mesure --series` et `--torsion-fixe` (sprint 22/11/2026). Sans carte : T1 à T5. Avec carte (`--ignored`) : G1 à G4.

use kvu::commun::{BackendKind, Depart, Graine, Precision};
use kvu::gpu::{self, Appareil};
use kvu::su2::torsion::Torsion;
use kvu::su2::une_boite::{tables_boite, tables_boite_fixe, ParamBoite, UneBoite};
use kvu::su2::ParamChaine;
use std::process::Command;

fn pb(cotes: [usize; 4], tors: &str, poids: f64, ch: Vec<ParamChaine>) -> ParamBoite {
    ParamBoite {
        cotes,
        torsion: Torsion::analyser(tors).unwrap(),
        poids,
        nor: 2,
        chaines: ch,
        precision: Precision::F64,
        impairs: false,
        aniso: None,
        couture: None,
    }
}
fn tors(t: &str) -> Torsion {
    Torsion::analyser(t).unwrap()
}
fn un() -> Vec<ParamChaine> {
    vec![ParamChaine { beta: 2.4, graine: Graine(7), depart: Depart::Froid }]
}
fn bits(c: &Option<Vec<f64>>) -> Vec<u64> {
    c.as_ref().unwrap().iter().map(|x| x.to_bits()).collect()
}

/// T1 : identité 1 de la spécification (tables) : fixe 01 + variable 02 à s = −1 = torsion 01,02 à s = −1, coefficients au bit.
#[test]
fn t1_identite_1_coefficients() {
    for c in [[4, 4, 4, 4], [4, 6, 4, 8]] {
        let (_, a, _) = tables_boite_fixe(&pb(c, "02", -1.0, un()), Some(&tors("01"))).unwrap();
        let (_, b, _) = tables_boite(&pb(c, "01,02", -1.0, un())).unwrap();
        assert_eq!(bits(&a), bits(&b), "{c:?}");
    }
}

/// T2 : identité 2 : fixe 01 + variable 02 à s = +1 = torsion 01 à s = −1, au bit.
#[test]
fn t2_identite_2_coefficients() {
    for c in [[4, 4, 4, 4], [4, 6, 4, 8]] {
        let (_, a, _) = tables_boite_fixe(&pb(c, "02", 1.0, un()), Some(&tors("01"))).unwrap();
        let (_, b, _) = tables_boite(&pb(c, "01", -1.0, un())).unwrap();
        assert_eq!(bits(&a), bits(&b), "{c:?}");
    }
}

/// T3 : le masque de feuille ne contient que la pile à poids variable ; fixe absent ou vide = tables_boite, au bit ; refus d'un plan
/// dans les deux listes.
#[test]
fn t3_masque_refus_et_neutralite() {
    let c = [4, 4, 4, 4];
    let v = 256;
    let (_, coef, fe) = tables_boite_fixe(&pb(c, "02", 0.3, un()), Some(&tors("01"))).unwrap();
    assert_eq!(fe[..v].iter().filter(|&&x| x != 0).count(), 0, "plan 01 hors feuille");
    assert_eq!(fe[v..2 * v].iter().filter(|&&x| x != 0).count(), 16, "plan 02 : pile de 16 plaquettes (4^2 positions des deux autres coordonnées)");
    let coef = coef.unwrap();
    assert_eq!(coef.iter().filter(|&&x| x == -1.0).count(), 16);
    assert_eq!(coef.iter().filter(|&&x| x == 0.3).count(), 16);
    for t in ["aucune", "01,23"] {
        let p = pb(c, t, 0.3, un());
        let (_, a, fa) = tables_boite(&p).unwrap();
        for f in [None, Some(Torsion::triviale())] {
            let (_, b, fb) = tables_boite_fixe(&p, f.as_ref()).unwrap();
            assert_eq!(a.as_ref().map(|x| bits(&Some(x.clone()))), b.as_ref().map(|x| bits(&Some(x.clone()))));
            assert_eq!(fa, fb);
        }
    }
    let e = tables_boite_fixe(&pb(c, "01,02", -1.0, un()), Some(&tors("02"))).unwrap_err();
    assert!(e.to_string().contains("02"), "{e}");
}

/// T4 (sans carte) : refus en ligne de commande, avant tout accès à la carte.
#[test]
fn t4_refus_cli() {
    let o = Command::new(env!("CARGO_BIN_EXE_kvu"))
        .args(["su2", "thermo", "mesure", "--cotes", "4,4,4,4", "--torsion", "01,02", "--torsion-fixe", "02", "--betas", "2.4",
               "--repliques", "2", "--therm", "2", "--nmeas", "2", "--sortie", "/nonexistent/x"])
        .output()
        .unwrap();
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--torsion-fixe"), "{}", String::from_utf8_lossy(&o.stderr));
}

// ---------------------------------------------------------------- avec carte
fn appareil() -> Appareil {
    let b = if cfg!(feature = "cuda") { BackendKind::Cuda } else { BackendKind::Hip };
    gpu::ouvrir(b, 0).unwrap_or_else(|e| panic!("FAIL : backend {} indisponible ({e})", b.nom()))
}
fn serie(app: &Appareil, p: &ParamBoite, fixe: Option<&Torsion>) -> Vec<Vec<[f64; 2]>> {
    let mut m = match fixe {
        Some(f) => UneBoite::nouveau_fixe(app, p, f).unwrap(),
        None => UneBoite::nouveau(app, p).unwrap(),
    };
    m.init().unwrap();
    m.serie(5, 6, 1).unwrap()
}
fn deux() -> Vec<ParamChaine> {
    vec![
        ParamChaine { beta: 2.4, graine: Graine(11), depart: Depart::Froid },
        ParamChaine { beta: 2.4, graine: Graine(12), depart: Depart::Chaud },
    ]
}
fn eq_bits(a: &[Vec<[f64; 2]>], b: &[Vec<[f64; 2]>], k: usize) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        for (u, v) in x.iter().zip(y) {
            assert_eq!(u[k].to_bits(), v[k].to_bits());
        }
    }
}

/// G1 : identités 1 et 2 sur les séries de la carte, S_tot au bit (S_F diffère par construction : pile à poids variable seule).
#[test]
#[ignore = "GPU"]
fn g1_identites_au_bit_sur_carte() {
    let app = appareil();
    let c = [4, 4, 4, 4];
    let a = serie(&app, &pb(c, "02", -1.0, deux()), Some(&tors("01")));
    let b = serie(&app, &pb(c, "01,02", -1.0, deux()), None);
    eq_bits(&a, &b, 0);
    // identité 1 : S_F(variable 02) + S_F(pile 01) = S_F(deux piles) ; la pile 01 seule vient de l'identité 2 à s = +1 / fixe 02
    let a1 = serie(&app, &pb(c, "01", -1.0, deux()), Some(&tors("02")));
    for ((x, y), z) in a.iter().zip(&a1).zip(&b) {
        for ((u, v), w) in x.iter().zip(y).zip(z) {
            assert!((u[1] + v[1] - w[1]).abs() < 1e-11, "S_F : {} + {} contre {}", u[1], v[1], w[1]);
        }
    }
    let d = serie(&app, &pb(c, "02", 1.0, deux()), Some(&tors("01")));
    let e = serie(&app, &pb(c, "01", -1.0, deux()), None);
    eq_bits(&d, &e, 0);
    println!("PASS G1");
}

// ---- référence hôte (indexation propre au test) : S_F de la pile du plan p au coin (l_μ−1, l_ν−1), S_tot brute
fn hote(d: [usize; 4], q: &[f64], p: usize) -> (f64, f64) {
    use kvu::heritage::su2_hb::{qdag, qmul, Q};
    const PL: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
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
    let li = |m: usize, s: usize| -> Q {
        let i = (m * v + s) * 4;
        [q[i], q[i + 1], q[i + 2], q[i + 3]]
    };
    let (mut tot, mut fe) = (0.0, 0.0);
    for s in 0..v {
        let x = co(s);
        for (pp, &(mu, nu)) in PL.iter().enumerate() {
            let a = qmul(li(mu, s), li(nu, hu(x, mu)));
            let b = qmul(qdag(li(mu, hu(x, nu))), qdag(li(nu, s)));
            let val = qmul(a, b)[0];
            tot += val;
            if pp == p && x[mu] == d[mu] - 1 && x[nu] == d[nu] - 1 {
                fe += val;
            }
        }
    }
    (tot, fe)
}

/// G2 : processeur contre carte : S_tot et S_F (pile 02 seule, 01 fixe) d'un champ imposé, à 1e-11 relatif.
#[test]
#[ignore = "GPU"]
fn g2_hote_egale_carte_avec_pile_fixe() {
    let app = appareil();
    let d = [4, 4, 4, 4];
    let v = 256;
    let mut z = 99u64;
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
    let mut m = UneBoite::nouveau_fixe(&app, &pb(d, "02", -1.0, un()), &tors("01")).unwrap();
    m.init().unwrap();
    m.televerser(0, &q).unwrap();
    m.allouer_historique(1).unwrap();
    m.mesurer(0).unwrap();
    let h = m.historique().unwrap();
    let (tot, fe) = hote(d, &q, 1);
    assert!((h[0] - tot).abs() < 1e-11 * (1.0 + tot.abs()), "S_tot {} / {tot}", h[0]);
    assert!((h[1] - fe).abs() < 1e-11 * (1.0 + fe.abs()), "S_F {} / {fe}", h[1]);
    println!("PASS G2");
}

fn kvu(args: &[&str]) -> (i32, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_kvu")).args(args).env("KEVINOTRON_VERROU", "ignorer").output().unwrap();
    let mut s = String::from_utf8_lossy(&o.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&o.stderr));
    (o.status.code().unwrap_or(-1), s)
}

/// G3 : `--series` absent : points/*.csv identiques à ceux de `--series` présent ; moyenne des séries = colonnes du csv, exactement.
#[test]
#[ignore = "GPU"]
fn g3_cli_series() {
    let b = if cfg!(feature = "cuda") { "cuda" } else { "hip" };
    let base = std::env::temp_dir().join(format!("kvu_series_g3_{}", std::process::id()));
    let (s0, s1) = (base.join("sans"), base.join("avec"));
    let com = |o: &str, ex: &[&str]| {
        let mut a = vec!["su2", "thermo", "mesure", "--cotes", "4,4,4,4", "--torsion-fixe", "01", "--torsion", "02", "--poids", "-1,1",
                         "--betas", "2.3,2.4", "--repliques", "3", "--therm", "5", "--nmeas", "8", "--backend", b, "--sortie", o];
        a.extend_from_slice(ex);
        let (c, t) = kvu(&a);
        assert_eq!(c, 0, "{t}");
    };
    com(s0.to_str().unwrap(), &[]);
    com(s1.to_str().unwrap(), &["--series"]);
    for ia in 0..2 {
        let f = format!("points/bras{ia}_g000.csv");
        let c0 = std::fs::read(s0.join(&f)).unwrap();
        assert_eq!(c0, std::fs::read(s1.join(&f)).unwrap(), "csv {f}");
        let csv = String::from_utf8(c0).unwrap();
        let rd = |k: &str| -> Vec<f64> {
            let o = std::fs::read(s1.join(format!("series/bras{ia}_g000_{k}.bin"))).unwrap();
            o.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
        };
        let (fe, tot) = (rd("fe"), rd("tot"));
        assert_eq!(fe.len(), 8 * 6);
        for (ch, ln) in csv.lines().skip(1).enumerate() {
            let c: Vec<&str> = ln.split(',').collect();
            for (k, ser) in [(6usize, &tot), (7, &fe)] {
                let mut x = 0.0;
                for t in 0..8 {
                    x += ser[t * 6 + ch];
                }
                assert_eq!((x / 8.0).to_bits(), c[k].parse::<f64>().unwrap().to_bits(), "bras {ia} chaîne {ch} col {k}");
            }
        }
    }
    assert!(!s0.join("series").exists());
    let man = std::fs::read_to_string(s1.join("manifest_thermo.json")).unwrap();
    assert!(man.contains("\"series\"") && man.contains("torsion_fixe"));
    let _ = std::fs::remove_dir_all(&base);
    println!("PASS G3");
}

fn verifier_lecteur_g4(lecteur: &std::path::Path) {
    assert!(lecteur.is_file(), "G4 : lecteur Python manquant : {}", lecteur.display());
}

#[test]
#[should_panic(expected = "G4 : lecteur Python manquant")]
fn t5_lecteur_g4_absent() {
    let absent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml/lire_series.py");
    verifier_lecteur_g4(&absent);
}

/// G4 (dette D de la fusion n°4) : PLUSIEURS groupes de β avec `--series`. 9 β, 2 répliques, `--chaines-max 10` : 5 β par groupe, donc un
/// groupe de 5 β (10 chaînes) et un groupe de 4 β (8 chaînes). Côté écriture de kvu : un couple `_fe.bin`/`_tot.bin` par bras et par
/// groupe, taille = nmeas × chaînes du groupe × 8 octets ; la moyenne de chaque colonne des séries redonne EXACTEMENT la colonne du
/// csv du groupe ; le manifeste donne `betas_par_groupe` = 5. Puis, si python3 et numpy sont disponibles, le lecteur
/// `scripts/thermo/lire_series.py` relit un nœud de chaque groupe (rang dans le groupe et indice du groupe corrects).
#[test]
#[ignore = "GPU"]
fn g4_cli_series_deux_groupes_de_beta() {
    let lecteur = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/thermo/lire_series.py");
    verifier_lecteur_g4(&lecteur);
    let b = if cfg!(feature = "cuda") { "cuda" } else { "hip" };
    let base = std::env::temp_dir().join(format!("kvu_series_g4_{}", std::process::id()));
    let s = base.join("deux_groupes");
    let (r, nm) = (2usize, 6usize);
    let betas = "2.20,2.25,2.30,2.35,2.40,2.45,2.50,2.55,2.60";
    let (c, t) = kvu(&[
        "su2", "thermo", "mesure", "--cotes", "4,4,4,4", "--torsion-fixe", "01", "--torsion", "02", "--poids", "-1,1", "--betas", betas,
        "--repliques", "2", "--chaines-max", "10", "--therm", "5", "--nmeas", "6", "--backend", b, "--series", "--sortie",
        s.to_str().unwrap(),
    ]);
    assert_eq!(c, 0, "{t}");
    let man: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(s.join("manifest_thermo.json")).unwrap()).unwrap();
    assert_eq!(man["series"]["betas_par_groupe"].as_u64(), Some(5), "{man}");
    assert_eq!(man["series"]["nmeas"].as_u64(), Some(nm as u64));
    assert_eq!(man["series"]["repliques"].as_u64(), Some(r as u64));
    // pas de troisième groupe
    for ia in 0..2 {
        assert!(!s.join(format!("series/bras{ia}_g002_fe.bin")).exists(), "troisième groupe inattendu");
        assert!(!s.join(format!("points/bras{ia}_g002.csv")).exists(), "troisième groupe inattendu");
    }
    for ia in 0..2 {
        for (g, nbeta) in [(0usize, 5usize), (1, 4)] {
            let ncol = nbeta * r;
            let csv = std::fs::read_to_string(s.join(format!("points/bras{ia}_g{g:03}.csv"))).unwrap();
            assert_eq!(csv.lines().count(), 1 + ncol, "lignes du csv bras {ia} groupe {g}");
            for (k, nom) in [(6usize, "tot"), (7, "fe")] {
                let o = std::fs::read(s.join(format!("series/bras{ia}_g{g:03}_{nom}.bin"))).unwrap();
                assert_eq!(o.len(), 8 * nm * ncol, "taille de bras{ia}_g{g:03}_{nom}.bin");
                let ser: Vec<f64> = o.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
                assert!(ser.iter().all(|x| x.is_finite()));
                for (ch, ln) in csv.lines().skip(1).enumerate() {
                    let col: Vec<&str> = ln.split(',').collect();
                    let mut x = 0.0;
                    for t in 0..nm {
                        x += ser[t * ncol + ch];
                    }
                    assert_eq!(
                        (x / nm as f64).to_bits(),
                        col[k].parse::<f64>().unwrap().to_bits(),
                        "bras {ia} groupe {g} chaîne {ch} col {k} ({nom})"
                    );
                }
            }
        }
    }
    // relecture par le lecteur python (un nœud du groupe 0, un nœud du groupe 1) ; sauté sans python3 + numpy
    let numpy = Command::new("python3").args(["-I", "-c", "import numpy"]).output();
    if numpy.map(|o| o.status.success()).unwrap_or(false) {
        for (beta, g, k) in [("2.30", 0u64, 2u64), ("2.60", 1, 3)] {
            let o = Command::new("python3")
                .args(["-I", "-B", lecteur.to_str().unwrap(), s.to_str().unwrap(), "--beta", beta, "--bootstrap", "0"])
                .output()
                .unwrap();
            let rc = o.status.code().unwrap_or(-1);
            assert!(rc == 0 || rc == 3, "lecteur beta {beta} : rc {rc}\n{}", String::from_utf8_lossy(&o.stderr));
            let j: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
            assert_eq!(j["groupe"].as_u64(), Some(g), "{j}");
            assert_eq!(j["rang_dans_groupe"].as_u64(), Some(k), "{j}");
            assert_eq!(j["R"].as_u64(), Some(r as u64));
            assert_eq!(j["nmeas"].as_u64(), Some(nm as u64));
        }
        println!("PASS G4 relecture Python");
    } else {
        eprintln!("G4 : SAUT VISIBLE de la relecture Python (python3 ou NumPy indisponible) : seul le producteur est attesté");
        println!("PASS G4 PRODUCTEUR SEUL (lecteur SAUTE)");
        let _ = std::fs::remove_dir_all(&base);
        return;
    }
    let _ = std::fs::remove_dir_all(&base);
    println!("PASS G4");
}
