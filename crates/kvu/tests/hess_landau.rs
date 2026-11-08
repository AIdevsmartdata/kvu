//! Fixation de Landau (port de lat.py::landau_fix) : jauge pure, critère, indépendance au nombre de fils, entrées invalides.
use kvu::commun::ChampHote;
use kvu::hess::bases::Reseau;
use kvu::hess::landau::{critere, fixer};
use kvu::heritage::su2_hb::{qdag, qmul};
use rayon::ThreadPoolBuilder;

struct Alea(u64);
impl Alea {
    fn suivant(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        2.0 * ((z >> 11) as f64 / 9_007_199_254_740_992.0) - 1.0
    }
    fn quaternion(&mut self) -> [f64; 4] {
        loop {
            let c = [self.suivant(), self.suivant(), self.suivant(), self.suivant()];
            let n2: f64 = c.iter().map(|x| x * x).sum();
            if n2 > 1e-6 && n2 <= 1.0 {
                let n = n2.sqrt();
                return c.map(|x| x / n);
            }
        }
    }
}

fn haar(l: usize, graine: u64) -> ChampHote {
    let mut a = Alea(graine);
    let q: Vec<f64> = (0..4 * l.pow(4)).flat_map(|_| a.quaternion()).collect();
    ChampHote { l, q }
}

fn jauge_pure(l: usize, graine: u64) -> ChampHote {
    let r = Reseau::nouveau(l).unwrap();
    let mut a = Alea(graine);
    let omega: Vec<[f64; 4]> = (0..r.v).map(|_| a.quaternion()).collect();
    let mut q = vec![0.0; 16 * r.v];
    for mu in 0..4 {
        for x in 0..r.v {
            let u = qmul(omega[x], qdag(omega[r.avant[mu][x]]));
            q[4 * (mu * r.v + x)..4 * (mu * r.v + x) + 4].copy_from_slice(&u);
        }
    }
    ChampHote { l, q }
}

fn pool(n: usize) -> rayon::ThreadPool {
    ThreadPoolBuilder::new().num_threads(n).build().unwrap()
}

#[test]
fn jauge_pure_devient_un_champ_constant() {
    let c = jauge_pure(4, 7);
    let r = Reseau::nouveau(4).unwrap();
    assert!(critere(&r, &c.q) > 1e-3, "le fond de jauge pure n'est pas en jauge de Landau");
    let s = fixer(&pool(4), &c, 1e-24, 5000, 1.7).unwrap();
    assert!(s.converge, "θ = {:e} après {} itérations", s.theta, s.iterations);
    // le critère est nul et tous les liens d'une même direction sont égaux (le fond est équivalent à U = 1 à une rotation globale près)
    let base = &s.champ.q[0..4];
    for lien in s.champ.q.chunks_exact(4) {
        for c in 0..4 {
            assert!((lien[c] - base[c]).abs() < 1e-9, "lien non constant : {lien:?} contre {base:?}");
        }
    }
}

#[test]
fn critere_decroit_et_unitarite_conservee() {
    let c = haar(4, 11);
    let r = Reseau::nouveau(4).unwrap();
    let avant = critere(&r, &c.q);
    let s = fixer(&pool(2), &c, 1e-20, 400, 1.7).unwrap();
    assert!(s.theta < avant * 1e-6, "θ : {avant:e} → {:e}", s.theta);
    assert!(s.champ.defaut_unitarite() < 1e-13);
}

#[test]
fn independant_du_nombre_de_fils() {
    let c = haar(4, 13);
    let a = fixer(&pool(1), &c, 1e-24, 1000, 1.7).unwrap();
    for n in [3, 8] {
        let b = fixer(&pool(n), &c, 1e-24, 1000, 1.7).unwrap();
        assert_eq!(a.iterations, b.iterations);
        assert!(a.champ.q.iter().zip(&b.champ.q).all(|(x, y)| x.to_bits() == y.to_bits()), "octets différents à {n} fils");
    }
}

#[test]
fn entrees_invalides() {
    let mut c = haar(2, 1);
    c.q.pop();
    assert!(fixer(&pool(1), &c, 1e-20, 10, 1.7).is_err());
    assert!(Reseau::nouveau(1).is_err());
}
