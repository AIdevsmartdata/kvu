#![forbid(unsafe_code)]

use kvu::hess::hv::{GeoPlaq, MutantHv, OperateurGauche};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::Arc;
use std::time::Instant;

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

    fn indice(&mut self, taille: usize) -> usize {
        (self.entier() % taille as u64) as usize
    }
}

fn pool(fils: usize) -> ThreadPool {
    ThreadPoolBuilder::new()
        .num_threads(fils)
        .build()
        .expect("création du pool")
}

fn configuration(l: usize, graine: u64) -> Vec<f64> {
    let mut alea = Aleatoire(graine);
    let mut q = Vec::with_capacity(16 * l.pow(4));
    for _ in 0..4 * l.pow(4) {
        // La projection radiale d'un point uniforme de la boule donne Haar sur S³.
        loop {
            let candidat = [alea.reel(), alea.reel(), alea.reel(), alea.reel()];
            let norme2: f64 = candidat.iter().map(|x| x * x).sum();
            if norme2 > 0.0 && norme2 <= 1.0 {
                let norme = norme2.sqrt();
                q.extend(candidat.map(|x| x / norme));
                break;
            }
        }
    }
    q
}

fn vecteur(n: usize, graine: u64) -> Vec<f64> {
    let mut alea = Aleatoire(graine);
    (0..n).map(|_| alea.reel()).collect()
}

// Calcul indépendant de la bibliothèque, en composantes scalaires.
fn multiplier(a: &[f64], b: &[f64]) -> [f64; 4] {
    let scalaire = a[0] * b[0] - a[1..].iter().zip(&b[1..]).map(|(x, y)| x * y).sum::<f64>();
    let mut resultat = [scalaire, 0.0, 0.0, 0.0];
    for couleur in 0..3 {
        let suivante = (couleur + 1) % 3 + 1;
        let derniere = (couleur + 2) % 3 + 1;
        resultat[couleur + 1] = a[0] * b[couleur + 1] + b[0] * a[couleur + 1]
            - a[suivante] * b[derniere]
            + a[derniere] * b[suivante];
    }
    resultat
}

fn perturber(q: &[f64], composante: usize, t: f64) -> Vec<f64> {
    let e = composante / 3;
    let a = composante % 3;
    let angle = t * std::f64::consts::FRAC_1_SQRT_2;
    let mut exponentielle = [angle.cos(), 0.0, 0.0, 0.0];
    exponentielle[a + 1] = angle.sin();
    let produit = multiplier(&exponentielle, &q[4 * e..4 * e + 4]);
    let mut resultat = q.to_vec();
    resultat[4 * e..4 * e + 4].copy_from_slice(&produit);
    resultat
}

fn derivee_action(pool: &ThreadPool, q: &[f64], l: usize, i: usize, h: f64) -> f64 {
    let action = |t| {
        OperateurGauche::nouveau(pool, &perturber(q, i, t), l)
            .expect("configuration perturbée")
            .action()
    };
    let grossiere = (action(h) - action(-h)) / (2.0 * h);
    let fine = (action(h / 2.0) - action(-h / 2.0)) / h;
    (4.0 * fine - grossiere) / 3.0
}

fn derivee_gradient(pool: &ThreadPool, q: &[f64], l: usize, i: usize, h: f64) -> Vec<f64> {
    let gradient = |t| {
        let operateur = OperateurGauche::nouveau(pool, &perturber(q, i, t), l)
            .expect("configuration perturbée");
        let mut g = vec![f64::NAN; operateur.n()];
        operateur.gradient(pool, &mut g);
        g
    };
    let plus = gradient(h);
    let moins = gradient(-h);
    let demi_plus = gradient(h / 2.0);
    let demi_moins = gradient(-h / 2.0);
    (0..plus.len())
        .map(|j| {
            let grossiere = (plus[j] - moins[j]) / (2.0 * h);
            let fine = (demi_plus[j] - demi_moins[j]) / h;
            (4.0 * fine - grossiere) / 3.0
        })
        .collect()
}

// Énumération géométrique indépendante, via coordonnées explicites et pas toriques.
fn plaquettes(l: usize) -> Vec<[usize; 4]> {
    let volume = l.pow(4);
    let indice = |x: [usize; 4]| ((x[0] * l + x[1]) * l + x[2]) * l + x[3];
    let mut resultat = Vec::new();
    for mu in 0..4 {
        for nu in mu + 1..4 {
            for x0 in 0..l {
                for x1 in 0..l {
                    for x2 in 0..l {
                        for x3 in 0..l {
                            let x = [x0, x1, x2, x3];
                            let mut x_mu = x;
                            let mut x_nu = x;
                            x_mu[mu] = (x_mu[mu] + 1) % l;
                            x_nu[nu] = (x_nu[nu] + 1) % l;
                            resultat.push([
                                mu * volume + indice(x),
                                nu * volume + indice(x_mu),
                                mu * volume + indice(x_nu),
                                nu * volume + indice(x),
                            ]);
                        }
                    }
                }
            }
        }
    }
    resultat
}

fn scalaires_boucles(q: &[f64], l: usize) -> Vec<f64> {
    plaquettes(l)
        .iter()
        .map(|liens| {
            let mut produit = [1.0, 0.0, 0.0, 0.0];
            for (s, &e) in liens.iter().enumerate() {
                let mut facteur = [q[4 * e], q[4 * e + 1], q[4 * e + 2], q[4 * e + 3]];
                if s >= 2 {
                    facteur[1..].iter_mut().for_each(|x| *x = -*x);
                }
                produit = multiplier(&produit, &facteur);
            }
            produit[0]
        })
        .collect()
}

fn produit_scalaire(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norme(a: &[f64]) -> f64 {
    produit_scalaire(a, a).sqrt()
}

fn appliquer(
    operateur: &OperateurGauche,
    pool: &ThreadPool,
    v: &[f64],
    mutant: MutantHv,
) -> Vec<f64> {
    let mut y = vec![f64::NAN; operateur.n()];
    operateur.hv(pool, v, &mut y, mutant);
    y
}

// Pour H symétrique, la colonne calculée fournit aussi la somme absolue de la ligne.
fn schur_estime(operateur: &OperateurGauche, pool: &ThreadPool, lignes: &[usize]) -> f64 {
    let mut maximum: f64 = 0.0;
    let mut base = vec![0.0; operateur.n()];
    for &i in lignes {
        base[i] = 1.0;
        let ligne = appliquer(operateur, pool, &base, MutantHv::Aucun);
        maximum = maximum.max(ligne.iter().map(|x| x.abs()).sum());
        base[i] = 0.0;
    }
    maximum
}

#[test]
fn gradient_contre_differences_finies() {
    let pool = pool(4);
    let l = 4;
    let q = configuration(l, 0x381e_ba73);
    let operateur = OperateurGauche::nouveau(&pool, &q, l).expect("configuration valide");
    assert_eq!(operateur.l(), l);
    assert_eq!(operateur.n(), 12 * l.pow(4));
    let mut g = vec![f64::NAN; operateur.n()];
    operateur.gradient(&pool, &mut g);
    let mut alea = Aleatoire(0x6732_bf91);
    let mut maximum: f64 = 0.0;
    for _ in 0..12 {
        let i = alea.indice(g.len());
        let derivee = derivee_action(&pool, &q, l, i, 0.03);
        let ecart = (derivee - g[i]).abs() / derivee.abs().max(g[i].abs());
        maximum = maximum.max(ecart);
        assert!(
            ecart <= 1e-8,
            "composante {i} : gradient {}, dérivée {derivee}, écart {ecart:e}",
            g[i]
        );
    }
    println!("test 1 : écart relatif maximal = {maximum:.6e}");
}

#[test]
fn hessienne_contre_differences_finies_du_gradient() {
    let pool = pool(4);
    let l = 4;
    let q = configuration(l, 0x4ae1_7163);
    let operateur = OperateurGauche::nouveau(&pool, &q, l).expect("configuration valide");
    let boucles = plaquettes(l);
    let mut alea = Aleatoire(0x8973_ae15);
    let mut couples = Vec::new();
    for k in 0..8 {
        let liens = boucles[alea.indice(boucles.len())];
        let i = 3 * liens[0] + alea.indice(3);
        // Même couleur sur le même lien : dérivée ordinaire = Hessienne symétrisée.
        // Les autres couples portent sur des liens distincts, dont les champs commutent.
        let j = if k < 4 {
            i
        } else {
            3 * liens[1 + k % 3] + alea.indice(3)
        };
        couples.push((i, j));
    }
    let lignes: Vec<_> = couples.iter().map(|&(_, j)| j).collect();
    let schur = schur_estime(&operateur, &pool, &lignes);
    let mut maximum: f64 = 0.0;
    let mut base = vec![0.0; operateur.n()];
    for (i, j) in couples {
        base[i] = 1.0;
        let hv = appliquer(&operateur, &pool, &base, MutantHv::Aucun);
        let derivee = derivee_gradient(&pool, &q, l, i, 0.002);
        let ecart = (hv[j] - derivee[j]).abs() / schur;
        maximum = maximum.max(ecart);
        assert!(
            ecart <= 1e-7,
            "couple ({i}, {j}) : H {}, dérivée {}, écart {ecart:e}",
            hv[j],
            derivee[j]
        );
        base[i] = 0.0;
    }
    println!("test 2 : écart maximal / Schur = {maximum:.6e}, Schur estimé = {schur:.6e}");
}

#[test]
fn symetrie() {
    let pool = pool(4);
    for l in [2_usize, 4] {
        let q = configuration(l, 0x319f_7b32);
        let operateur = OperateurGauche::nouveau(&pool, &q, l).expect("configuration valide");
        let v = vecteur(operateur.n(), 0x7893_afb1);
        let w = vecteur(operateur.n(), 0x142b_8f31);
        let hv = appliquer(&operateur, &pool, &v, MutantHv::Aucun);
        let hw = appliquer(&operateur, &pool, &w, MutantHv::Aucun);
        let schur = schur_estime(
            &operateur,
            &pool,
            &[0, operateur.n() / 2, operateur.n() - 1],
        );
        let ecart = (produit_scalaire(&w, &hv) - produit_scalaire(&v, &hw)).abs();
        assert!(
            ecart <= 1e-12 * norme(&v) * norme(&w) * schur,
            "L={l}, écart {ecart:e}"
        );
    }
}

#[test]
fn froid() {
    let pool = pool(4);
    let mut maximum: f64 = 0.0;
    for l in [2_usize, 4] {
        let mut q = vec![0.0; 16 * l.pow(4)];
        for lien in q.chunks_mut(4) {
            lien[0] = 1.0;
        }
        let operateur = OperateurGauche::nouveau(&pool, &q, l).expect("configuration froide");
        assert_eq!(operateur.action(), 0.0);
        let mut gradient = vec![f64::NAN; operateur.n()];
        operateur.gradient(&pool, &mut gradient);
        assert!(gradient.iter().all(|&x| x == 0.0));
        for graine in [0x6432_ae32, 0x4138_7fe1, 0xa317_bf43] {
            let v = vecteur(operateur.n(), graine);
            let hv = appliquer(&operateur, &pool, &v, MutantHv::Aucun);
            let mut attendu = vec![0.0; operateur.n()];
            for liens in plaquettes(l) {
                for a in 0..3 {
                    let rotation = v[3 * liens[0] + a] + v[3 * liens[1] + a]
                        - v[3 * liens[2] + a]
                        - v[3 * liens[3] + a];
                    for (s, e) in liens.into_iter().enumerate() {
                        attendu[3 * e + a] += if s < 2 {
                            0.5 * rotation
                        } else {
                            -0.5 * rotation
                        };
                    }
                }
            }
            let erreur = hv
                .iter()
                .zip(&attendu)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max);
            let echelle = attendu.iter().map(|x| x.abs()).fold(0.0, f64::max);
            let ecart = erreur / echelle;
            maximum = maximum.max(ecart);
            assert!(ecart <= 1e-13, "L={l}, écart {ecart:e}");
        }
    }
    println!("test 4 : écart relatif maximal = {maximum:.6e}");
}

fn memes_bits(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "composante {i}");
    }
}

#[test]
fn independance_aux_fils() {
    let debut = Instant::now();
    let pools: Vec<_> = [1, 3, 4, 8].into_iter().map(pool).collect();
    for l in [2, 4, 6] {
        let q = configuration(l, 0x418a_3f72);
        let operateur = OperateurGauche::nouveau(&pools[0], &q, l).expect("configuration valide");
        let v = vecteur(operateur.n(), 0xf783_614a);
        let hv = appliquer(&operateur, &pools[0], &v, MutantHv::Aucun);
        let mutant = appliquer(&operateur, &pools[0], &v, MutantHv::MemeLienRetire);
        let mut g = vec![f64::NAN; operateur.n()];
        operateur.gradient(&pools[0], &mut g);
        for pool in &pools {
            let autre = OperateurGauche::nouveau(pool, &q, l).expect("configuration valide");
            assert_eq!(operateur.action().to_bits(), autre.action().to_bits());
            memes_bits(&hv, &appliquer(&autre, pool, &v, MutantHv::Aucun));
            memes_bits(
                &mutant,
                &appliquer(&autre, pool, &v, MutantHv::MemeLienRetire),
            );
            let mut gradient = vec![f64::NAN; operateur.n()];
            autre.gradient(pool, &mut gradient);
            memes_bits(&g, &gradient);
            // Vérifie aussi le changement de pool après construction du cache.
            memes_bits(&hv, &appliquer(&operateur, pool, &v, MutantHv::Aucun));
        }
        assert!(Arc::ptr_eq(&GeoPlaq::pour(l), &GeoPlaq::pour(l)));
    }
    println!(
        "indépendance aux fils : L=2,4,6 ; 1,3,4,8 fils ; durée {:?}",
        debut.elapsed()
    );
}

#[test]
fn mutant_meme_lien() {
    let pool = pool(4);
    let l = 4;
    let q = configuration(l, 0x13ef_7b15);
    let operateur = OperateurGauche::nouveau(&pool, &q, l).expect("configuration valide");
    let v = vecteur(operateur.n(), 0x4173_afb1);
    let hv = appliquer(&operateur, &pool, &v, MutantHv::Aucun);
    let mutant = appliquer(&operateur, &pool, &v, MutantHv::MemeLienRetire);
    let scalaires = scalaires_boucles(&q, l);
    let mut diagonales = vec![0.0; operateur.n() / 3];
    for (liens, u0) in plaquettes(l).iter().zip(scalaires) {
        for &e in liens {
            diagonales[e] += 0.5 * u0;
        }
    }
    let mut maximum: f64 = 0.0;
    for i in 0..v.len() {
        // Retirer la diagonale impose H_correct − H_mutant = + diagonale × v.
        let attendu = diagonales[i / 3] * v[i];
        let erreur = (hv[i] - mutant[i] - attendu).abs();
        maximum = maximum.max(erreur);
        assert!(erreur <= 1e-13, "composante {i}, écart {erreur:e}");
    }
    let action0 = operateur.action();
    let mut base = vec![0.0; operateur.n()];
    let mut maximum_correct: f64 = 0.0;
    let mut minimum_mutant = f64::INFINITY;
    // Évite une courbure presque nulle, mal conditionnée pour une erreur relative.
    let liens: Vec<_> = diagonales
        .iter()
        .enumerate()
        .filter(|(_, d)| d.abs() > 0.3)
        .step_by(17)
        .take(5)
        .map(|(e, _)| e)
        .collect();
    assert_eq!(liens.len(), 5);
    for (couleur, e) in liens.into_iter().enumerate() {
        let i = 3 * e + couleur % 3;
        base[i] = 1.0;
        let courbure = |h: f64| {
            let plus = OperateurGauche::nouveau(&pool, &perturber(&q, i, h), l)
                .expect("perturbation positive")
                .action();
            let moins = OperateurGauche::nouveau(&pool, &perturber(&q, i, -h), l)
                .expect("perturbation négative")
                .action();
            (plus - 2.0 * action0 + moins) / (h * h)
        };
        let h = 0.05;
        let derivee = (4.0 * courbure(h / 2.0) - courbure(h)) / 3.0;
        let correct = appliquer(&operateur, &pool, &base, MutantHv::Aucun)[i];
        let retire = appliquer(&operateur, &pool, &base, MutantHv::MemeLienRetire)[i];
        let ecart_correct = (correct - derivee).abs() / derivee.abs();
        let ecart_mutant = (retire - derivee).abs() / derivee.abs();
        maximum_correct = maximum_correct.max(ecart_correct);
        minimum_mutant = minimum_mutant.min(ecart_mutant);
        assert!(
            ecart_correct <= 1e-6,
            "lien {e}, écart correct {ecart_correct:e}"
        );
        assert!(
            ecart_mutant >= 0.1,
            "lien {e}, écart mutant {ecart_mutant:e}"
        );
        base[i] = 0.0;
    }
    println!("mutant : écart diagonal maximal = {maximum:.6e}, erreur correcte maximale = {maximum_correct:.6e}, erreur mutante minimale = {minimum_mutant:.6e}");
}

#[test]
fn entrees_invalides() {
    let pool = pool(1);
    for l in [0, 1, usize::MAX] {
        let erreur = OperateurGauche::nouveau(&pool, &[], l)
            .err()
            .expect("taille rejetée");
        assert!(!erreur.is_empty());
    }
    for longueur in [0, 16 * 2_usize.pow(4) - 1, 16 * 2_usize.pow(4) + 1] {
        assert!(OperateurGauche::nouveau(&pool, &vec![0.0; longueur], 2).is_err());
    }
    // Aucune vérification d'unitarité ne fait partie de l'API.
    assert!(OperateurGauche::nouveau(&pool, &vec![0.0; 16 * 2_usize.pow(4)], 2).is_ok());
}
