//! Références indépendantes : spectres analytiques, réflexions et résidus explicites.
use kvu::hess::dense::*;
use kvu::hess::lobpcg::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::f64::consts::PI;

struct Alea(u64);
impl Alea {
    fn suivant(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        2.0 * (((z ^ (z >> 31)) >> 11) as f64 / 9007199254740992.0) - 1.0
    }
}
fn pool(fils: usize) -> ThreadPool {
    ThreadPoolBuilder::new().num_threads(fils).build().unwrap()
}
fn identite(n: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        a[i * n + i] = 1.0;
    }
    a
}
fn orthogonale(n: usize, graine: u64) -> Vec<f64> {
    let mut q = identite(n);
    let mut alea = Alea(graine);
    // Produit explicite de réflexions indépendantes du solveur testé.
    for _ in 0..12 {
        let mut v: Vec<_> = (0..n).map(|_| alea.suivant()).collect();
        let norme = norme2(None, &v);
        for x in &mut v {
            *x /= norme;
        }
        for ligne in q.chunks_mut(n) {
            let facteur = 2.0 * pdscal(None, ligne, &v);
            for (x, &u) in ligne.iter_mut().zip(&v) {
                *x -= facteur * u;
            }
        }
    }
    q
}
fn spectrale(q: &[f64], valeurs: &[f64]) -> Vec<f64> {
    let n = valeurs.len();
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut somme = 0.0;
            for k in 0..n {
                somme += q[i * n + k] * valeurs[k] * q[j * n + k];
            }
            a[i * n + j] = somme;
            a[j * n + i] = somme;
        }
    }
    a
}
fn appliquer_dense(a: &[f64], x: &[f64]) -> Vec<f64> {
    a.chunks(x.len())
        .map(|ligne| pdscal(None, ligne, x))
        .collect()
}
fn erreur_orthogonale(cols: &[Vec<f64>]) -> f64 {
    let mut erreur: f64 = 0.0;
    for (i, x) in cols.iter().enumerate() {
        for (j, y) in cols.iter().enumerate() {
            erreur = erreur.max((pdscal(None, x, y) - f64::from(i == j)).abs());
        }
    }
    erreur
}
fn colonnes(v: &[f64], n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|j| (0..n).map(|i| v[i * n + j]).collect())
        .collect()
}
fn octets(x: &[f64]) -> Vec<u64> {
    x.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn cholesky_et_resolutions() {
    let n = 50;
    let q = orthogonale(n, 19);
    let a = spectrale(
        &q,
        &(0..n).map(|i| 0.2 + i as f64 / 9.0).collect::<Vec<_>>(),
    );
    let mut l = a.clone();
    cholesky(&mut l, n).unwrap();
    let mut erreur = Vec::new();
    for i in 0..n {
        for j in 0..n {
            erreur
                .push(pdscal(None, &l[i * n..(i + 1) * n], &l[j * n..(j + 1) * n]) - a[i * n + j]);
            if j > i {
                assert_eq!(l[i * n + j], 0.0);
            }
        }
    }
    assert!(norme2(None, &erreur) <= 1e-13 * norme2(None, &a));
    let x: Vec<_> = (0..n).map(|i| (i as f64).cos()).collect();
    let mut second = appliquer_dense(&a, &x);
    resoudre_l(&l, n, &mut second);
    resoudre_lt(&l, n, &mut second);
    assert!(
        norme2(
            None,
            &second.iter().zip(x).map(|(a, b)| a - b).collect::<Vec<_>>()
        ) < 1e-12
    );
    assert_eq!(cholesky(&mut [1.0, 2.0, 2.0, 1.0], 2), Err(1));
}

#[test]
fn eigh_sym_contre_references() {
    let mut erreur_valeurs: f64 = 0.0;
    let mut erreur_residus: f64 = 0.0;
    let mut erreur_orth: f64 = 0.0;
    for n in [60, 300, 400] {
        let (a, mut reference) = if n == 400 {
            let mut a = vec![0.0; n * n];
            for i in 0..n {
                a[i * n + i] = 2.0;
                a[i * n + (i + 1) % n] = -1.0;
                a[i * n + (i + n - 1) % n] = -1.0;
            }
            (
                a,
                (0..n)
                    .map(|k| 2.0 - 2.0 * (2.0 * PI * k as f64 / n as f64).cos())
                    .collect::<Vec<_>>(),
            )
        } else {
            let valeurs: Vec<_> = (0..n)
                .map(|i| -2.0 + 6.0 * i as f64 / (n - 1) as f64)
                .collect();
            (spectrale(&orthogonale(n, n as u64), &valeurs), valeurs)
        };
        reference.sort_by(f64::total_cmp);
        let (valeurs, v) = eigh_sym(a.clone(), n, true).unwrap();
        let seuls = eigh_sym(a.clone(), n, false).unwrap();
        assert_eq!(octets(&valeurs), octets(&seuls.0));
        assert!(seuls.1.is_none());
        let cols = colonnes(&v.unwrap(), n);
        let mut residus = Vec::new();
        for (j, x) in cols.iter().enumerate() {
            for (av, &x) in appliquer_dense(&a, x).iter().zip(x) {
                residus.push(av - valeurs[j] * x);
            }
        }
        let ev = valeurs
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let er = norme2(None, &residus) / norme2(None, &a);
        let eo = erreur_orthogonale(&cols);
        erreur_valeurs = erreur_valeurs.max(ev);
        erreur_residus = erreur_residus.max(er);
        erreur_orth = erreur_orth.max(eo);
        assert!(ev <= 4e-12, "n={n}, valeurs={ev:e}");
        assert!(er <= 1e-12, "n={n}, résidu={er:e}");
        assert!(eo <= 1e-13, "n={n}, orthogonalité={eo:e}");
    }
    eprintln!("MESURE test 2 : valeurs abs={erreur_valeurs:e}, résidu relatif Frobenius={erreur_residus:e}, orthogonalité max={erreur_orth:e}");
    assert!(eigh_sym(vec![], 0, true).unwrap().0.is_empty());
    assert_eq!(eigh_sym(vec![3.0], 1, true).unwrap().0, vec![3.0]);
}

#[test]
fn generalise() {
    let n = 80;
    let a = spectrale(
        &orthogonale(n, 31),
        &(0..n).map(|i| 0.2 + i as f64 / 20.0).collect::<Vec<_>>(),
    );
    let b = spectrale(
        &orthogonale(n, 41),
        &(0..n).map(|i| 0.5 + i as f64 / 40.0).collect::<Vec<_>>(),
    );
    let (valeurs, v) = eigh_generalise(a.clone(), b.clone(), n, true).unwrap();
    let cols = colonnes(&v.unwrap(), n);
    for (j, x) in cols.iter().enumerate() {
        let ax = appliquer_dense(&a, x);
        let bx = appliquer_dense(&b, x);
        let residu: Vec<_> = ax
            .iter()
            .zip(&bx)
            .map(|(a, b)| a - valeurs[j] * b)
            .collect();
        assert!(norme2(None, &residu) < 1e-12);
        for (i, y) in cols.iter().enumerate() {
            assert!((pdscal(None, y, &bx) - f64::from(i == j)).abs() < 1e-13);
        }
    }
    let reference: Vec<_> = (0..n).map(|i| -0.5 + i as f64 / 17.0).collect();
    let mut a = spectrale(&orthogonale(n, 53), &reference);
    let mut b = vec![0.0; n * n];
    for i in 0..n {
        let qi = 0.5 + i as f64 / n as f64;
        b[i * n + i] = qi;
        for j in 0..n {
            a[i * n + j] *= (qi * (0.5 + j as f64 / n as f64)).sqrt();
        }
    }
    let (valeurs, _) = eigh_generalise(a, b, n, false).unwrap();
    assert!(valeurs
        .iter()
        .zip(reference)
        .all(|(a, b)| (a - b).abs() < 1e-12));
}

fn donnees_svd(rang: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let m = 300;
    let n = 40;
    let u = orthogonale(m, 71);
    let v = orthogonale(n, 73);
    let sigma: Vec<_> = (0..n)
        .map(|j| {
            if j < rang {
                10.0_f64.powf(-9.0 * j as f64 / (rang - 1) as f64)
            } else {
                0.0
            }
        })
        .collect();
    let mut cols = vec![vec![0.0; m]; n];
    for j in 0..n {
        for i in 0..m {
            for k in 0..n {
                cols[j][i] += u[i * m + k] * sigma[k] * v[j * n + k];
            }
        }
    }
    (cols, sigma)
}

#[test]
fn svd_jacobi() {
    let pool = pool(4);
    let (mut cols, reference) = donnees_svd(40);
    let original = cols.clone();
    let (sigma, v) = svd_jacobi_colonnes(Some(&pool), &mut cols, true).unwrap();
    let mut triees = sigma.clone();
    triees.sort_by(|a, b| b.total_cmp(a));
    let mut erreur: f64 = 0.0;
    let mut relative: f64 = 0.0;
    for (s, r) in triees.iter().zip(reference) {
        erreur = erreur.max((s - r).abs());
        relative = relative.max((s - r).abs() / r);
        assert!(
            (s - r).abs() <= 1e-12 * r + 1e-13,
            "σ={r:e}, erreur={:e}",
            (s - r).abs()
        );
    }
    let v = v.unwrap();
    let mut reconstruction: f64 = 0.0;
    for j in 0..40 {
        for i in 0..300 {
            let mut valeur = 0.0;
            for k in 0..40 {
                valeur += cols[k][i] * v[j * 40 + k];
            }
            reconstruction = reconstruction.max((valeur - original[j][i]).abs());
        }
    }
    assert!(reconstruction < 1e-13);
    let (mut deficientes, _) = donnees_svd(25);
    let (sigma, _) = svd_jacobi_colonnes(Some(&pool), &mut deficientes, false).unwrap();
    let maximum = sigma.iter().copied().fold(0.0, f64::max);
    // colonnes nulles : ignorées sous 1e-13 σ_max par la SVD de Jacobi (leur norme finale n'est donc plus réduite à l'arrondi) ; seuil de rang bien plus haut (1e-10 σ_max)
    assert_eq!(sigma.iter().filter(|&&s| s < 2e-13 * maximum).count(), 15);
    let (original, _) = donnees_svd(25);
    let (base, sigma, ecartes) = base_de_l_image(Some(&pool), original.clone(), 1e-12).unwrap();
    assert_eq!(base.len(), 25);
    assert_eq!(ecartes, 15);
    assert!(sigma.windows(2).all(|s| s[0] >= s[1]));
    let orth = erreur_orthogonale(&base);
    assert!(orth <= 1e-13);
    let mut image: f64 = 0.0;
    for mut c in original {
        let coefficients: Vec<_> = base.iter().map(|u| pdscal(None, u, &c)).collect();
        for (u, coefficient) in base.iter().zip(coefficients) {
            for (x, y) in c.iter_mut().zip(u) {
                *x -= coefficient * y;
            }
        }
        image = image.max(norme2(None, &c));
    }
    assert!(image <= 1e-12);
    eprintln!("MESURE test 4 : σ abs={erreur:e}, σ relatif={relative:e}, reconstruction={reconstruction:e}, orthogonalité={orth:e}, image={image:e}");
}

#[test]
fn svd_independance_aux_fils() {
    let (original, _) = donnees_svd(40);
    let mut reference = None;
    for fils in [1, 3, 8] {
        let mut cols = original.clone();
        let (s, v) = svd_jacobi_colonnes(Some(&pool(fils)), &mut cols, true).unwrap();
        let octets = (
            octets(&s),
            cols.iter().map(|c| octets(c)).collect::<Vec<_>>(),
            octets(&v.unwrap()),
        );
        if let Some(reference) = &reference {
            assert_eq!(reference, &octets);
        } else {
            reference = Some(octets);
        }
    }
    // Plusieurs tronçons, dont un dernier incomplet.
    let mut alea = Alea(9);
    let x: Vec<_> = (0..20001).map(|_| alea.suivant()).collect();
    let y: Vec<_> = (0..20001).map(|_| alea.suivant()).collect();
    for fils in [1, 3, 8] {
        assert_eq!(
            pdscal(Some(&pool(fils)), &x, &y).to_bits(),
            pdscal(None, &x, &y).to_bits()
        );
    }
}

struct Identite(usize);
impl Op for Identite {
    fn n(&self) -> usize {
        self.0
    }
    fn appliquer(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }
}
struct Laplacien(usize);
impl Op for Laplacien {
    fn n(&self) -> usize {
        self.0
    }
    fn appliquer(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.0 {
            y[i] = 2.0 * x[i] - x[(i + 1) % self.0] - x[(i + self.0 - 1) % self.0];
        }
    }
}
struct HorsConstante(usize);
impl Op for HorsConstante {
    fn n(&self) -> usize {
        self.0
    }
    fn appliquer(&self, x: &[f64], y: &mut [f64]) {
        let constante = vec![1.0; x.len()];
        let moyenne = pdscal(None, x, &constante) / x.len() as f64;
        for (y, &x) in y.iter_mut().zip(x) {
            *y = x - moyenne;
        }
    }
}

#[test]
fn lobpcg_laplacien() {
    let n = 2000;
    let pool = pool(4);
    let a = Laplacien(n);
    let b = Identite(n);
    let hors = HorsConstante(n);
    for (proj, decale) in [(&b as &dyn Op, false), (&hors as &dyn Op, true)] {
        let p = Params {
            echelle: 4.0,
            iter_max: 4000,
            ..Params::default()
        };
        let s = lobpcg(&pool, &a, &b, proj, None, &p).unwrap();
        let mut reference: Vec<_> = (0..n)
            .map(|k| 2.0 - 2.0 * (2.0 * PI * k as f64 / n as f64).cos())
            .collect();
        reference.sort_by(f64::total_cmp);
        if decale {
            reference.remove(0);
        }
        let erreur = s
            .valeurs
            .iter()
            .zip(reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        eprintln!("MESURE test 6 : projeté={decale}, valeurs abs={erreur:e}, résidu relatif={:e}, bloc={:e}, itérations={}, convergé={}", s.residu_rel, s.historique.last().unwrap().1, s.iterations, s.converge);
        assert!(erreur < 1e-9);
        assert!(s.residu_rel <= 1e-9);
        assert!(s.converge);
        assert!((pdscal(None, &s.vecteur, &s.vecteur) - 1.0).abs() < 1e-13);
        if decale {
            assert!(pdscal(None, &s.vecteur, &vec![1.0; n]).abs() < 1e-12);
        }
        let av = projeter_test(proj, &projeter_test(&a, &s.vecteur));
        let r: Vec<_> = av
            .iter()
            .zip(&s.vecteur)
            .map(|(a, x)| a - s.valeurs[0] * x)
            .collect();
        assert!(norme2(None, &r) / 4.0 < 1e-9);
    }
}
fn projeter_test(op: &dyn Op, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    op.appliquer(x, &mut y);
    y
}

struct Pencil {
    n: usize,
    masse: bool,
}
impl Op for Pencil {
    fn n(&self) -> usize {
        self.n
    }
    fn appliquer(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.n {
            let q = 0.5 + 1.5 * ((37 * i % 103) as f64 / 102.0);
            y[i] = q * x[i];
            if !self.masse {
                y[i] += 0.018 * (i as f64 * 1.31).sin() * x[i];
                if i > 0 {
                    y[i] += 0.012 * x[i - 1];
                }
                if i + 1 < self.n {
                    y[i] += 0.012 * x[i + 1];
                }
            }
        }
    }
}
struct Projecteur {
    n: usize,
    exclus: usize,
}
impl Op for Projecteur {
    fn n(&self) -> usize {
        self.n
    }
    fn appliquer(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
        for i in 0..self.exclus {
            let composante = 0.5 * (x[2 * i] + x[2 * i + 1]);
            y[2 * i] -= composante;
            y[2 * i + 1] -= composante;
        }
    }
}
fn reference_projetee(n: usize, exclus: usize) -> Vec<f64> {
    let a = Pencil { n, masse: false };
    let b = Pencil { n, masse: true };
    let dim = n - exclus;
    // Base analytique du complément des (e_2i + e_(2i+1))/sqrt(2).
    let base: Vec<Vec<f64>> = (0..dim)
        .map(|j| {
            let mut x = vec![0.0; n];
            if j < exclus {
                x[2 * j] = 0.5_f64.sqrt();
                x[2 * j + 1] = -0.5_f64.sqrt();
            } else {
                x[j + exclus] = 1.0;
            }
            x
        })
        .collect();
    let mut ar = vec![0.0; dim * dim];
    let mut br = vec![0.0; dim * dim];
    for j in 0..dim {
        let ax = projeter_test(&a, &base[j]);
        let bx = projeter_test(&b, &base[j]);
        for i in 0..dim {
            ar[i * dim + j] = pdscal(None, &base[i], &ax);
            br[i * dim + j] = pdscal(None, &base[i], &bx);
        }
    }
    eigh_generalise(ar, br, dim, false).unwrap().0
}
fn resoudre_pencil(n: usize, cote: Cote, graine: u64, fils: usize) -> Sortie {
    lobpcg(
        &pool(fils),
        &Pencil { n, masse: false },
        &Pencil { n, masse: true },
        &Projecteur { n, exclus: 30 },
        None,
        &Params {
            cote,
            graine,
            echelle: 2.1,
            ..Params::default()
        },
    )
    .unwrap()
}

#[test]
fn lobpcg_generalise_projete() {
    let reference = reference_projetee(400, 30);
    let s = resoudre_pencil(400, Cote::Min, 11, 4);
    let erreur = s
        .valeurs
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    eprintln!("MESURE test 7 : contrôle n=400, valeurs abs={erreur:e}, résidu relatif={:e}, itérations={}", s.residu_rel, s.iterations);
    assert!(erreur < 1e-9);
    assert!(s.converge);
    let s = resoudre_pencil(3000, Cote::Min, 11, 4);
    let b = Pencil {
        n: 3000,
        masse: true,
    };
    let a = Pencil {
        n: 3000,
        masse: false,
    };
    let proj = Projecteur {
        n: 3000,
        exclus: 30,
    };
    let bx = projeter_test(&proj, &projeter_test(&b, &s.vecteur));
    assert!((pdscal(None, &s.vecteur, &bx) - 1.0).abs() < 1e-13);
    let ax = projeter_test(&proj, &projeter_test(&a, &s.vecteur));
    let r: Vec<_> = ax
        .iter()
        .zip(&bx)
        .map(|(a, b)| a - s.valeurs[0] * b)
        .collect();
    let residu = norme2(None, &r) / (2.1 * norme2(None, &s.vecteur));
    let px = projeter_test(&proj, &s.vecteur);
    let erreur_projection = s
        .vecteur
        .iter()
        .zip(px)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    eprintln!("MESURE test 7 : n=3000, résidu explicite={residu:e}, projection={erreur_projection:e}, itérations={}, convergé={}", s.iterations, s.converge);
    assert!(residu <= 1e-9);
    assert!(erreur_projection < 1e-13);
    assert!(s.converge);
}

#[test]
fn lobpcg_cote_max() {
    let reference = reference_projetee(400, 30);
    let s = resoudre_pencil(400, Cote::Max, 13, 4);
    let erreur = s
        .valeurs
        .iter()
        .zip(reference.iter().rev())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    eprintln!(
        "MESURE test 8 : valeurs abs={erreur:e}, résidu={:e}, itérations={}",
        s.residu_rel, s.iterations
    );
    assert!(erreur < 1e-9);
    assert!(s.converge);
}
type Signature = (Vec<u64>, Vec<u64>, u64, u64, usize, bool, Vec<(usize, u64)>);
fn signature(s: &Sortie) -> Signature {
    (
        octets(&s.valeurs),
        octets(&s.vecteur),
        s.residu_abs.to_bits(),
        s.residu_rel.to_bits(),
        s.iterations,
        s.converge,
        s.historique
            .iter()
            .map(|&(i, r)| (i, r.to_bits()))
            .collect(),
    )
}
#[test]
fn lobpcg_independance_aux_fils() {
    let mut reference = None;
    for fils in [1, 4, 8] {
        let s = resoudre_pencil(160, Cote::Min, 17, fils);
        assert!(s.converge);
        let octets = signature(&s);
        if let Some(reference) = &reference {
            assert_eq!(reference, &octets);
        } else {
            reference = Some(octets);
        }
    }
}
#[test]
fn lobpcg_graine() {
    let a = resoudre_pencil(160, Cote::Min, 19, 4);
    let b = resoudre_pencil(160, Cote::Min, 23, 4);
    let c = resoudre_pencil(160, Cote::Min, 19, 4);
    assert!(a.converge && b.converge);
    assert!(a
        .valeurs
        .iter()
        .zip(&b.valeurs)
        .all(|(a, b)| (a - b).abs() < 1e-9));
    assert_ne!(octets(&a.vecteur), octets(&b.vecteur));
    assert_eq!(signature(&a), signature(&c));
}
#[test]
fn lobpcg_entrees_invalides() {
    struct Oppose;
    impl Op for Oppose {
        fn n(&self) -> usize {
            10
        }
        fn appliquer(&self, x: &[f64], y: &mut [f64]) {
            for (y, x) in y.iter_mut().zip(x) {
                *y = -*x;
            }
        }
    }
    let pool = pool(1);
    let id = Identite(10);
    let p = Params::default();
    assert!(lobpcg(&pool, &id, &Oppose, &id, None, &p).is_err());
    for echelle in [0.0, -1.0, f64::NAN] {
        assert!(lobpcg(&pool, &id, &id, &id, None, &Params { echelle, ..p }).is_err());
    }
    assert!(lobpcg(&pool, &id, &id, &id, None, &Params { bloc: 0, ..p }).is_err());
    let sans = lobpcg(&pool, &Laplacien(10), &id, &id, None, &p).unwrap();
    let avec = lobpcg(&pool, &Laplacien(10), &id, &id, Some(&id), &p).unwrap();
    assert!(sans.converge && avec.converge);
    assert_eq!(signature(&sans), signature(&avec));
    let s = lobpcg(
        &pool,
        &Laplacien(10),
        &id,
        &id,
        None,
        &Params { iter_max: 0, ..p },
    )
    .unwrap();
    assert!(!s.converge);
    assert_eq!(s.iterations, 0);
}
