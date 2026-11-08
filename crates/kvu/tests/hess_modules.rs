#![forbid(unsafe_code)]

use kvu::commun::{ChampHote, Erreur};
use kvu::heritage::su2_hb::{qdag, qmul};
use kvu::hess::bases::{Base, Coord, Reseau};
use kvu::hess::dual::{jacobien_dense, jvp, support_lien};
use kvu::hess::espaces::{construire, Espace, ModeProj};
use kvu::hess::hamilt::{lmin, Hessienne, MutantHess, OpRef, OptionsLmin, Reference};
use kvu::hess::lobpcg::Op;
use kvu::hess::ops::{self, adjointe, q0_appliquer, QCov};
use kvu::rg::bloc::{bloquer, OptionsBloc};
use kvu::rg::geometrie::{Geometrie, Table};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::Arc;
use std::time::Instant;

struct Duree(&'static str, Instant);
impl Duree {
    fn nouvelle(nom: &'static str) -> Self {
        Self(nom, Instant::now())
    }
}
impl Drop for Duree {
    fn drop(&mut self) {
        eprintln!("{} : {:.3} s", self.0, self.1.elapsed().as_secs_f64());
    }
}

// SplitMix64 et tirage Haar repris de hess_hv.rs.
struct Aleatoire(u64);
impl Aleatoire {
    fn entier(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn reel(&mut self) -> f64 {
        2.0 * ((self.entier() >> 11) as f64 / 9_007_199_254_740_992.0) - 1.0
    }
    fn quaternion(&mut self) -> [f64; 4] {
        loop {
            let q = [self.reel(), self.reel(), self.reel(), self.reel()];
            let n2: f64 = q.iter().map(|x| x * x).sum();
            if n2 > 0.0 && n2 <= 1.0 {
                return q.map(|x| x / n2.sqrt());
            }
        }
    }
}
fn pool(fils: usize) -> Arc<ThreadPool> {
    Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(fils)
            .build()
            .expect("création du pool"),
    )
}
fn configuration(l: usize, graine: u64) -> ChampHote {
    let mut alea = Aleatoire(graine);
    let q = (0..4 * l.pow(4)).flat_map(|_| alea.quaternion()).collect();
    ChampHote { l, q }
}
fn froid(l: usize) -> ChampHote {
    ChampHote {
        l,
        q: (0..4 * l.pow(4))
            .flat_map(|_| [1.0, 0.0, 0.0, 0.0])
            .collect(),
    }
}
fn vecteur(n: usize, graine: u64) -> Vec<f64> {
    let mut alea = Aleatoire(graine);
    (0..n).map(|_| alea.reel()).collect()
}
fn scalaire(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "longueurs des vecteurs");
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}
fn norme(x: &[f64]) -> f64 {
    scalaire(x, x).sqrt()
}
fn difference(x: &[f64], y: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), y.len(), "longueurs des vecteurs");
    x.iter().zip(y).map(|(a, b)| a - b).collect()
}
fn proche(x: &[f64], y: &[f64], tol: f64, contexte: &str) {
    let erreur = norme(&difference(x, y)) / norme(y).max(f64::MIN_POSITIVE);
    assert!(
        erreur <= tol,
        "{contexte} : écart relatif {erreur:e}, seuil {tol:e}"
    );
}
fn bits(x: &[f64], y: &[f64]) {
    assert_eq!(x.len(), y.len(), "longueurs des vecteurs");
    for (i, (a, b)) in x.iter().zip(y).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "composante {i}");
    }
}
fn appliquer(op: &dyn Op, x: &[f64]) -> Vec<f64> {
    let mut y = vec![f64::NAN; op.n()];
    op.appliquer(x, &mut y);
    y
}
fn hessienne(
    c: &ChampHote,
    p: Arc<ThreadPool>,
    base: Base,
    coord: Coord,
    mutant: MutantHess,
) -> Hessienne {
    Hessienne::nouveau(c, p, base, coord, 2.3, mutant).expect("construction de la Hessienne")
}
fn options(espace: Espace) -> OptionsLmin {
    OptionsLmin {
        espace,
        reference: if espace == Espace::Tcov {
            Reference::QCov
        } else {
            Reference::Q0
        },
        coord: if matches!(espace, Espace::Tcov | Espace::KerDu) {
            Coord::Gauche
        } else {
            Coord::Additive
        },
        beta: 2.3,
        graine: 1,
        tol: 1e-11,
        bloc: 4,
        iter_max: 3000,
        mode_proj: ModeProj::Cg,
        dense: false,
        mutant: MutantHess::Aucun,
        verifier_q: true,
    }
}
fn transformer(c: &ChampHote, omega: &[[f64; 4]]) -> ChampHote {
    let r = Reseau::nouveau(c.l).expect("réseau");
    let mut q = vec![0.0; c.q.len()];
    for x in 0..r.v {
        for mu in 0..4 {
            let e = mu * r.v + x;
            let u = qmul(qmul(omega[x], c.lien(e)), qdag(omega[r.avant[mu][x]]));
            q[4 * e..4 * e + 4].copy_from_slice(&u);
        }
    }
    ChampHote { l: c.l, q }
}

#[test]
fn q0_symetrique_positive_et_noyau() {
    let _d = Duree::nouvelle("q0_symetrique_positive_et_noyau");
    let r = Reseau::nouveau(4).expect("réseau");
    let plaq = r.plaquettes_s2();
    let q = |x: &[f64]| {
        let mut y = vec![f64::NAN; r.n];
        q0_appliquer(&plaq, x, &mut y);
        y
    };
    let x = vecteur(r.n, 11);
    let y = vecteur(r.n, 12);
    let (qx, qy) = (q(&x), q(&y));
    assert!(
        (scalaire(&x, &qy) - scalaire(&y, &qx)).abs() <= 1e-13 * norme(&x) * norme(&qy),
        "symétrie de Q0"
    );
    assert!(
        scalaire(&x, &qx) > 0.0 && scalaire(&y, &qy) > 0.0,
        "positivité de Q0"
    );
    let lambda = vecteur(3 * r.v, 13);
    let mut gradient = vec![0.0; r.n];
    for x in 0..r.v {
        for mu in 0..4 {
            for a in 0..3 {
                gradient[3 * (4 * x + mu) + a] = lambda[3 * r.avant[mu][x] + a] - lambda[3 * x + a];
            }
        }
    }
    assert!(
        norme(&q(&gradient)) <= 1e-12 * norme(&gradient),
        "gradient pur dans le noyau"
    );
}

#[test]
fn qcov_a_froid_egale_q0_et_covariance() {
    let _d = Duree::nouvelle("qcov_a_froid_egale_q0_et_covariance");
    let r = Reseau::nouveau(4).expect("réseau");
    let z = vecteur(r.n, 21);
    let w = vecteur(r.n, 22);
    let eval = |c: &ChampHote, x: &[f64]| {
        let mut y = vec![f64::NAN; r.n];
        QCov::nouveau(&r, &ops::liens_s2(c, &r)).appliquer(x, &mut y);
        y
    };
    let mut q0 = vec![0.0; r.n];
    q0_appliquer(&r.plaquettes_s2(), &z, &mut q0);
    proche(&eval(&froid(4), &z), &q0, 1e-13, "QCov froid = Q0");
    let c = configuration(4, 23);
    let (qz, qw) = (eval(&c, &z), eval(&c, &w));
    assert!(
        (scalaire(&z, &qw) - scalaire(&w, &qz)).abs() <= 1e-13 * norme(&z) * norme(&qw),
        "symétrie de QCov"
    );
    assert!(
        scalaire(&z, &qz) > 0.0 && scalaire(&w, &qw) > 0.0,
        "positivité de QCov"
    );
    let mut alea = Aleatoire(24);
    let omega: Vec<_> = (0..r.v).map(|_| alea.quaternion()).collect();
    let transforme = transformer(&c, &omega);
    let mut zp = z.clone();
    for (x, &o) in omega.iter().enumerate() {
        let rotation = adjointe(o);
        for mu in 0..4 {
            let debut = 3 * (4 * x + mu);
            for (a, ligne) in rotation.iter().enumerate() {
                zp[debut + a] = scalaire(ligne, &z[debut..debut + 3]);
            }
        }
    }
    let avant = scalaire(&z, &qz);
    let apres = scalaire(&zp, &eval(&transforme, &zp));
    let erreur = (apres - avant).abs() / avant.abs();
    assert!(erreur <= 1e-12, "covariance de jauge : {erreur:e}");
}

#[test]
fn projecteurs_svd_et_cg_concordent() {
    let _d = Duree::nouvelle("projecteurs_svd_et_cg_concordent");
    let c = configuration(4, 31);
    // Les trois SVD sont indépendantes ; leurs phases séquentielles se recouvrent.
    std::thread::scope(|portee| {
        for espace in [Espace::Libre, Espace::KerB, Espace::Tcov] {
            let c = &c;
            portee.spawn(move || {
                let debut = Instant::now();
                let p = pool(1);
                let svd = construire(espace, c, p.clone(), ModeProj::Svd).expect("projecteur SVD");
                let cg = construire(espace, c, p.clone(), ModeProj::Cg).expect("projecteur CG");
                if espace == Espace::Libre {
                    assert_eq!(svd.dim, Some(2295), "dimension libre");
                }
                if espace == Espace::KerB {
                    assert_eq!(svd.dim, Some(2127), "dimension kerB");
                }
                assert_eq!(svd.proj.dim(), svd.dim, "dimension du projecteur");
                let landau = if espace == Espace::Libre {
                    Some(
                        construire(Espace::Landau, c, p.clone(), ModeProj::Cg)
                            .expect("projecteur Landau"),
                    )
                } else {
                    None
                };
                for graine in 32..35 {
                    let z = vecteur(12 * c.v(), graine);
                    let w = vecteur(z.len(), graine + 10);
                    let ps = appliquer(&svd.proj, &z);
                    let pc = appliquer(&cg.proj, &z);
                    proche(&pc, &ps, 1e-11, espace.nom());
                    for proj in [&svd.proj, &cg.proj] {
                        let pz = appliquer(proj, &z);
                        let ppz = appliquer(proj, &pz);
                        let erreur = norme(&difference(&ppz, &pz)) / norme(&z);
                        assert!(
                            erreur <= 1e-11,
                            "{} : vecteur déjà dans X, écart {erreur:e}",
                            espace.nom()
                        );
                        let pw = appliquer(proj, &w);
                        assert!(
                            (scalaire(&pz, &w) - scalaire(&z, &pw)).abs()
                                <= 1e-11 * norme(&z) * norme(&w),
                            "{} : symétrie",
                            espace.nom()
                        );
                    }
                    if let Some(l) = &landau {
                        proche(&appliquer(&l.proj, &z), &ps, 1e-12, "libre = landau");
                    }
                }
                eprintln!(
                    "projecteurs {} : {:.3} s",
                    espace.nom(),
                    debut.elapsed().as_secs_f64()
                );
            });
        }
    });
}

#[test]
fn dimension_kerdu_l3() {
    let _d = Duree::nouvelle("dimension_kerdu_l3");
    let p = pool(4);
    for c in [configuration(3, 41), froid(3)] {
        let e = construire(Espace::KerDu, &c, p.clone(), ModeProj::Cg).expect("espace kerdu");
        assert_eq!(e.dim, Some(720), "dimension kerdu à L=3");
        for graine in 42..47 {
            let v = appliquer(&e.proj, &vecteur(12 * c.v(), graine));
            let dv = jvp(&p, &c, &v).expect("JVP projeté");
            let erreur = norme(&dv.dubar_v) / norme(&v);
            assert!(erreur <= 1e-10, "kerdu : norme relative du JVP {erreur:e}");
        }
    }
}

fn bloc(c: &ChampHote) -> ChampHote {
    bloquer(
        c,
        &OptionsBloc {
            geometrie: Geometrie::TroisCentre,
            fils: 2,
            angle_so3: false,
        },
    )
    .expect("blocage de référence")
    .grossier
}
fn perturber(c: &ChampHote, v: &[f64], h: f64) -> ChampHote {
    let r = Reseau::nouveau(c.l).expect("réseau");
    let mut q = c.q.clone();
    for l in 0..r.nl {
        let s: [f64; 3] = std::array::from_fn(|a| 0.5 * h * v[3 * l + a]);
        let angle = norme(&s);
        let facteur = if angle == 0.0 {
            1.0
        } else {
            angle.sin() / angle
        };
        let exp = [angle.cos(), facteur * s[0], facteur * s[1], facteur * s[2]];
        let e = r.champ_de_s2(l);
        q[4 * e..4 * e + 4].copy_from_slice(&qmul(exp, c.lien(e)));
    }
    ChampHote { l: c.l, q }
}
fn deplacement(c: &ChampHote, origine: &ChampHote) -> Vec<f64> {
    let mut v = Vec::with_capacity(12 * c.v());
    // L=3 fin donne un seul site grossier : les deux dispositions coïncident.
    assert_eq!(c.l, 1, "un site grossier attendu");
    for e in 0..4 {
        let q = qmul(c.lien(e), qdag(origine.lien(e)));
        let n = norme(&q[1..]);
        let facteur = if n == 0.0 {
            2.0
        } else {
            2.0 * n.atan2(q[0]) / n
        };
        v.extend(q[1..].iter().map(|x| facteur * x));
    }
    v
}
#[test]
fn jvp_contre_differences_finies_de_rg_bloc() {
    let _d = Duree::nouvelle("jvp_contre_differences_finies_de_rg_bloc");
    let p = pool(2);
    for graine in 51..54 {
        let c = configuration(3, graine);
        let origine = bloc(&c);
        for direction in 61..64 {
            let v = vecteur(12 * c.v(), direction);
            let j = jvp(&p, &c, &v).expect("JVP");
            assert_eq!(j.ubar.l, origine.l, "taille grossière");
            assert!(
                j.ubar
                    .q
                    .iter()
                    .zip(&origine.q)
                    .all(|(x, y)| (x - y).abs() <= 1e-14),
                "Ū dual différent du blocage"
            );
            let centre = |h: f64| {
                let plus = deplacement(&bloc(&perturber(&c, &v, h)), &origine);
                let moins = deplacement(&bloc(&perturber(&c, &v, -h)), &origine);
                difference(&plus, &moins)
                    .into_iter()
                    .map(|x| x / (2.0 * h))
                    .collect::<Vec<_>>()
            };
            let (c0, c1, c2) = (centre(2.5e-4), centre(1.25e-4), centre(6.25e-5));
            let fd: Vec<_> = c0
                .iter()
                .zip(&c1)
                .zip(&c2)
                .map(|((a, b), c)| {
                    let r1 = (4.0 * b - a) / 3.0;
                    let r2 = (4.0 * c - b) / 3.0;
                    (16.0 * r2 - r1) / 15.0
                })
                .collect();
            let erreur = norme(&difference(&fd, &j.dubar_v)) / norme(&j.dubar_v);
            eprintln!("fond {graine}, direction {direction} : écart FD {erreur:e}");
            assert!(
                erreur <= 1e-8,
                "fond {graine}, direction {direction} : écart FD {erreur:e}"
            );
        }
    }
}

#[test]
fn jacobien_dense_egal_jvp() {
    let _d = Duree::nouvelle("jacobien_dense_egal_jvp");
    let c = configuration(3, 71);
    let p = pool(4);
    let r = Reseau::nouveau(3).expect("réseau");
    let j = jacobien_dense(&p, &c).expect("Jacobien dense");
    assert_eq!(j.len(), 12, "lignes grossières");
    assert!(j.iter().all(|ligne| ligne.len() == r.n), "colonnes fines");
    for graine in 72..75 {
        let v = vecteur(r.n, graine);
        let produit: Vec<_> = j.iter().map(|ligne| scalaire(ligne, &v)).collect();
        proche(
            &produit,
            &jvp(&p, &c, &v).expect("JVP").dubar_v,
            1e-12,
            "Jacobien × vecteur",
        );
    }
    let table = Table::nouvelle(Geometrie::TroisCentre);
    for mu in 0..4 {
        let support = support_lien(&table, 3, [0; 4], mu);
        assert!(support.len() < r.nl, "support strictement local");
        for e in 0..r.nl {
            if !support.contains(&e) {
                for a in 0..3 {
                    for b in 0..3 {
                        assert_eq!(
                            j[3 * mu + b][3 * r.s2_de_champ(e) + a],
                            0.0,
                            "colonne hors support"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn hessienne_additive_symetrique_et_coherente() {
    let _d = Duree::nouvelle("hessienne_additive_symetrique_et_coherente");
    let p = pool(4);
    for c in [configuration(2, 81), froid(2)] {
        let n = 12 * c.v();
        let mut matrices = Vec::new();
        for coord in [Coord::Additive, Coord::Gauche] {
            let h = hessienne(&c, p.clone(), Base::S2, coord, MutantHess::Aucun);
            let mut colonnes = Vec::with_capacity(n);
            for i in 0..n {
                let mut ei = vec![0.0; n];
                ei[i] = 1.0;
                colonnes.push(appliquer(&h, &ei));
            }
            let mut erreur: f64 = 0.0;
            let mut echelle: f64 = 0.0;
            for (i, colonne) in colonnes.iter().enumerate() {
                for (k, autre) in colonnes.iter().enumerate() {
                    erreur = erreur.max((colonne[k] - autre[i]).abs());
                    echelle = echelle.max(colonne[k].abs());
                }
            }
            assert!(
                erreur <= 1e-12 * echelle,
                "symétrie {coord:?} : {erreur:e}, échelle {echelle:e}"
            );
            matrices.push(colonnes);
        }
        if c.q[0] == 1.0 {
            for (a, g) in matrices[0].iter().zip(&matrices[1]) {
                proche(a, g, 1e-13, "additive = gauche à froid");
            }
        }
        let hy = hessienne(&c, p.clone(), Base::Ym4, Coord::Gauche, MutantHess::Aucun);
        for i in 0..n {
            let mut ei = vec![0.0; n];
            let e = hy.reseau.champ_de_s2(i / 3);
            ei[3 * e + i % 3] = 1.0;
            let mut sortie = vec![0.0; n];
            hy.reseau.ym4_vers_s2(&appliquer(&hy, &ei), &mut sortie);
            let attendu: Vec<_> = matrices[1][i].iter().map(|x| 2.0 * x).collect();
            proche(&sortie, &attendu, 1e-13, "H ym4 = 2 H s2");
        }
    }
}

#[test]
fn lmin_froid_vaut_un() {
    let _d = Duree::nouvelle("lmin_froid_vaut_un");
    let p = pool(4);
    for espace in [
        Espace::Libre,
        Espace::KerB,
        Espace::Landau,
        Espace::Tcov,
        Espace::KerDu,
    ] {
        let c = froid(if espace == Espace::KerDu { 3 } else { 4 });
        let s = lmin(&c, p.clone(), &options(espace)).expect("minimum à froid");
        assert!(
            (s.lambda_min - 1.0).abs() <= 1e-9,
            "{} : λ_min = {}",
            espace.nom(),
            s.lambda_min
        );
    }
}

#[test]
fn lmin_dense_et_lobpcg_concordent() {
    let _d = Duree::nouvelle("lmin_dense_et_lobpcg_concordent");
    let c = configuration(2, 91);
    let p = pool(4);
    for espace in [Espace::Libre, Espace::KerB, Espace::Tcov] {
        let mode = if espace == Espace::Tcov {
            ModeProj::Svd
        } else {
            ModeProj::Cg
        };
        let e = construire(espace, &c, p.clone(), mode).expect("projecteur");
        if espace == Espace::Tcov && e.dim == Some(0) {
            eprintln!("tcov à L=2 : espace vide, comparaison des minima sautée");
            continue;
        }
        for beta in [2.3, 8.0] {
            let mut o = options(espace);
            o.beta = beta;
            o.mode_proj = mode;
            o.dense = true;
            let dense = lmin(&c, p.clone(), &o).expect("minimum dense");
            o.dense = false;
            let iteratif = lmin(&c, p.clone(), &o).expect("minimum LOBPCG");
            assert!(
                (dense.lambda_min - iteratif.lambda_min).abs() <= 1e-8,
                "{} β={beta} : dense {}, LOBPCG {}",
                espace.nom(),
                dense.lambda_min,
                iteratif.lambda_min
            );
            for s in [&dense, &iteratif] {
                proche(
                    &appliquer(&e.proj, &s.vecteur),
                    &s.vecteur,
                    1e-9,
                    "vecteur propre dans X",
                );
                let h = Hessienne::nouveau(&c, p.clone(), Base::S2, o.coord, beta, o.mutant)
                    .expect("Hessienne");
                let q = OpRef::nouveau(&c, &o.reference, beta).expect("référence");
                let hv = appliquer(&h, &s.vecteur);
                let qv = appliquer(&q, &s.vecteur);
                let residu: Vec<_> = hv
                    .iter()
                    .zip(&qv)
                    .map(|(h, q)| h - s.lambda_min * q)
                    .collect();
                let relatif = norme(&appliquer(&e.proj, &residu))
                    / (norme(&hv) + s.lambda_min.abs() * norme(&qv));
                assert!(relatif <= 1e-9, "résidu projeté explicite {relatif:e}");
                assert!(
                    s.json["residu_projete_rel"].as_f64().expect("résidu JSON") <= 1e-9,
                    "résidu du rapport"
                );
            }
        }
    }
}

#[test]
fn jauge_pure_tcov_et_mutant_mc() {
    let _d = Duree::nouvelle("jauge_pure_tcov_et_mutant_mc");
    let mut alea = Aleatoire(101);
    let omega: Vec<_> = (0..4_usize.pow(4)).map(|_| alea.quaternion()).collect();
    let c = transformer(&froid(4), &omega);
    let p = pool(4);
    let cov = lmin(&c, p.clone(), &options(Espace::Tcov)).expect("minimum covariant de jauge pure");
    assert!(cov.eps_prime.abs() <= 1e-9, "tcov : ε′ = {}", cov.eps_prime);
    // Pour distinguer |ε′| ≥ 1e-3, un résidu de 1e-7 suffit ; la convergence
    // du solveur est exigée plutôt qu'un arrêt à sa limite d'itérations.
    let mut o = options(Espace::Landau);
    o.tol = 1e-7;
    let landau = lmin(&c, p.clone(), &o).expect("mutant M-c sans fixation de Landau");
    assert_eq!(
        landau.json["solveur"]["converge"], true,
        "convergence du témoin M-c"
    );
    eprintln!(
        "jauge pure : ε′ tcov = {:.6e}, M-c = {:.6e}, itérations M-c = {}, résidu M-c = {}",
        cov.eps_prime,
        landau.eps_prime,
        landau.json["solveur"]["iterations"],
        landau.json["residu_projete_rel"]
    );
    assert!(
        landau.eps_prime.abs() >= 1e-3,
        "M-c : ε′ = {}",
        landau.eps_prime
    );
    let h = hessienne(&c, p.clone(), Base::S2, Coord::Gauche, MutantHess::Aucun);
    let mutant = hessienne(&c, p, Base::S2, Coord::Gauche, MutantHess::MemeLien);
    let mut ei = vec![0.0; h.n()];
    ei[0] = 1.0;
    let hv = appliquer(&h, &ei);
    let erreur = norme(&difference(&appliquer(&mutant, &ei), &hv)) / norme(&hv);
    assert!(erreur >= 0.1, "mutant même lien : écart {erreur:e}");
}

#[test]
fn lmin_independant_des_fils() {
    let _d = Duree::nouvelle("lmin_independant_des_fils");
    let c = configuration(4, 110);
    let o = options(Espace::Libre);
    let reference = lmin(&c, pool(1), &o).expect("minimum de référence");
    for fils in [3, 8] {
        let s = lmin(&c, pool(fils), &o).expect("minimum selon les fils");
        assert_eq!(
            reference.lambda_min.to_bits(),
            s.lambda_min.to_bits(),
            "λ_min selon les fils"
        );
        bits(&reference.vecteur, &s.vecteur);
        assert_eq!(reference.json, s.json, "JSON complet selon les fils");
        assert_eq!(
            serde_json::to_vec(&reference.json).expect("JSON"),
            serde_json::to_vec(&s.json).expect("JSON"),
            "octets JSON"
        );
    }
    let c = configuration(4, 111);
    let v = vecteur(12 * c.v(), 112);
    let h = hessienne(&c, pool(1), Base::S2, Coord::Additive, MutantHess::Aucun);
    let hv = appliquer(&h, &v);
    for fils in [3, 8] {
        let h = hessienne(&c, pool(fils), Base::S2, Coord::Additive, MutantHess::Aucun);
        bits(&hv, &appliquer(&h, &v));
    }
    let c = configuration(3, 113);
    let v = vecteur(12 * c.v(), 114);
    let a = jvp(&pool(1), &c, &v).expect("JVP un fil");
    let b = jvp(&pool(8), &c, &v).expect("JVP huit fils");
    bits(&a.dubar_v, &b.dubar_v);
    bits(&a.ubar.q, &b.ubar.q);
}

#[test]
fn espace_refuse_les_entrees_invalides() {
    let _d = Duree::nouvelle("espace_refuse_les_entrees_invalides");
    let p = pool(1);
    for l in [1, 3] {
        for espace in [Espace::KerB, Espace::Tcov] {
            assert!(
                matches!(
                    construire(espace, &froid(l), p.clone(), ModeProj::Cg),
                    Err(Erreur::Config(_))
                ),
                "{} : L={l} refusé",
                espace.nom()
            );
        }
    }
    assert!(
        matches!(
            construire(Espace::KerDu, &froid(4), p.clone(), ModeProj::Cg),
            Err(Erreur::Config(_))
        ),
        "kerdu : L non multiple de trois"
    );
    assert!(
        matches!(Reseau::nouveau(1), Err(Erreur::Config(_))),
        "réseau trop petit"
    );
    for longueur in [0, 971, 973] {
        assert!(
            matches!(
                jvp(&p, &froid(3), &vec![0.0; longueur]),
                Err(Erreur::Config(_))
            ),
            "JVP : longueur {longueur} refusée"
        );
    }
}
