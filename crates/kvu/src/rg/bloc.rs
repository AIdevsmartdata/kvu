//! Moyenne non linéaire S1 en quaternions SU(2), avec sommation séquentielle.

use super::geometrie::{Geometrie, Pas, Table, SITES_BLOC};
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::su2_hb::{qdag, qmul};
use std::f64::consts::PI;

pub type Q = [f64; 4];
pub const N_CLASSES: usize = 64;
const IDENTITE: Q = [1.0, 0.0, 0.0, 0.0];

pub fn log_q(w: Q) -> ([f64; 3], f64, bool) {
    let norme = w[1].hypot(w[2]).hypot(w[3]);
    let theta = norme.atan2(w[0]);
    let facteur = if norme < 1e-6 {
        let n2 = norme * norme;
        1.0 + n2 / 6.0 + 7.0 * n2 * n2 / 360.0
    } else {
        theta / norme
    };
    (
        [facteur * w[1], facteur * w[2], facteur * w[3]],
        theta,
        PI - theta < 1e-6,
    )
}

pub fn exp_q(s: [f64; 3]) -> Q {
    let norme = s[0].hypot(s[1]).hypot(s[2]);
    let facteur = if norme < 1e-6 {
        let n2 = norme * norme;
        1.0 - n2 / 6.0 + n2 * n2 / 120.0
    } else {
        norme.sin() / norme
    };
    [norme.cos(), facteur * s[0], facteur * s[1], facteur * s[2]]
}

#[derive(Clone, Debug, PartialEq)]
pub struct StatsTheta {
    pub n: u64,
    pub histogramme: [u64; N_CLASSES],
    pub n_hors: u64,
    pub n_ambigues: u64,
    pub theta_max: f64,
}

impl StatsTheta {
    pub fn vide() -> Self {
        Self {
            n: 0,
            histogramme: [0; N_CLASSES],
            n_hors: 0,
            n_ambigues: 0,
            theta_max: 0.0,
        }
    }

    pub fn ajouter(&mut self, theta: f64, ambigue: bool, angle_so3: bool) {
        let angle = if angle_so3 { 2.0 * theta } else { theta };
        let classe = ((angle * N_CLASSES as f64 / PI).floor() as usize).min(N_CLASSES - 1);
        self.n += 1;
        self.histogramme[classe] += 1;
        self.n_hors += u64::from(angle > PI / 2.0);
        self.n_ambigues += u64::from(ambigue);
        self.theta_max = self.theta_max.max(angle);
    }

    pub fn fusionner(&mut self, autre: &StatsTheta) {
        self.n += autre.n;
        for (a, b) in self.histogramme.iter_mut().zip(autre.histogramme) {
            *a += b;
        }
        self.n_hors += autre.n_hors;
        self.n_ambigues += autre.n_ambigues;
        self.theta_max = self.theta_max.max(autre.theta_max);
    }

    pub fn fraction_hors(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.n_hors as f64 / self.n as f64
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OptionsBloc {
    pub geometrie: Geometrie,
    pub fils: usize,
    pub angle_so3: bool,
}

pub struct SortieBloc {
    pub grossier: ChampHote,
    pub stats: StatsTheta,
}

/// Vérifie le format avant tout accès aux liens, sans imposer la normalisation.
pub(super) fn valider_champ(champ: &ChampHote) -> Resultat<()> {
    let taille = champ
        .l
        .checked_pow(4)
        .and_then(|v| v.checked_mul(16))
        .ok_or_else(|| Erreur::Config("taille du champ trop grande".into()))?;
    if champ.l == 0 || champ.l > (isize::MAX as usize).saturating_sub(8) || champ.q.len() != taille
    {
        return Err(Erreur::Config(
            "dimensions ou longueur du champ invalides".into(),
        ));
    }
    if let Some(i) = champ.q.iter().position(|x| !x.is_finite()) {
        return Err(Erreur::Config(format!(
            "composante non finie au lien {}, composante {}",
            i / 4,
            i % 4
        )));
    }
    Ok(())
}

pub(super) fn coordonnees(mut s: usize, l: usize) -> [usize; 4] {
    let mut x = [0; 4];
    for m in (0..4).rev() {
        x[m] = s % l;
        s /= l;
    }
    x
}

pub(super) fn site(x: [usize; 4], l: usize) -> usize {
    ((x[0] * l + x[1]) * l + x[2]) * l + x[3]
}

pub(super) fn deplacer(mut x: [usize; 4], mu: usize, delta: isize, l: usize) -> [usize; 4] {
    x[mu] = (x[mu] as isize + delta).rem_euclid(l as isize) as usize;
    x
}

pub(super) fn lire(champ: &ChampHote, x: [usize; 4], mu: usize) -> Q {
    champ.lien(mu * champ.v() + site(x, champ.l))
}

fn produit(champ: &ChampHote, origine: [usize; 4], chemin: &[Pas], translation: [i8; 4]) -> Q {
    let mut p = IDENTITE;
    for pas in chemin {
        let x = std::array::from_fn(|m| {
            (origine[m] as isize + pas.dep[m] as isize + translation[m] as isize)
                .rem_euclid(champ.l as isize) as usize
        });
        let u = lire(champ, x, pas.mu as usize);
        p = qmul(p, if pas.signe > 0 { u } else { qdag(u) });
    }
    p
}

/// Un seul tampon par fil, réutilisé pour tous ses liens et morceaux.
struct Tampon {
    depart: Vec<Q>,
    retour: Vec<Q>,
    segments: [Q; SITES_BLOC],
}

impl Tampon {
    fn nouveau(table: &Table) -> Self {
        Self {
            depart: vec![IDENTITE; table.nb_chemins()],
            retour: vec![IDENTITE; table.nb_chemins()],
            segments: [IDENTITE; SITES_BLOC],
        }
    }

    fn lien(
        &mut self,
        champ: &ChampHote,
        table: &Table,
        z: [usize; 4],
        mu: usize,
        angle_so3: bool,
    ) -> (Q, StatsTheta) {
        let origine = z.map(|v| 3 * v);
        let mut translation = [0; 4];
        translation[mu] = 3;
        for (i, p) in table.chemins.iter().enumerate() {
            self.depart[i] = produit(champ, origine, p, [0; 4]);
            self.retour[i] = qdag(produit(champ, origine, p, translation));
        }
        for (ir, segment) in self.segments.iter_mut().enumerate() {
            *segment = produit(champ, origine, &table.segment(mu, ir), [0; 4]);
        }
        let u0 = self.segments[Table::ir_de(table.ref_site)];
        let inverse = qdag(u0);
        let mut s = [0.0; 3];
        let mut stats = StatsTheta::vide();
        for ir in 0..SITES_BLOC {
            let poids = table.poids(ir);
            let debut = table.debut[ir];
            let fin = table.debut[ir + 1];
            for i in debut..fin {
                let pre = qmul(qmul(inverse, self.depart[i]), self.segments[ir]);
                for j in debut..fin {
                    let (ell, theta, ambigue) = log_q(qmul(pre, self.retour[j]));
                    for m in 0..3 {
                        s[m] += poids * ell[m];
                    }
                    stats.ajouter(theta, ambigue, angle_so3);
                }
            }
        }
        (qmul(u0, exp_q(s)), stats)
    }
}

/// Calcule un lien ; le champ, la direction et les coordonnées doivent être valides.
pub fn ubar_lien(champ: &ChampHote, table: &Table, z: [usize; 4], mu: usize) -> (Q, StatsTheta) {
    Tampon::nouveau(table).lien(champ, table, z, mu, false)
}

pub fn bloquer(champ: &ChampHote, opt: &OptionsBloc) -> Resultat<SortieBloc> {
    if champ.l < 3 || !champ.l.is_multiple_of(3) {
        return Err(Erreur::Config(format!(
            "L = {} : multiple de 3 ≥ 3 requis",
            champ.l
        )));
    }
    if opt.fils == 0 {
        return Err(Erreur::Config("le nombre de fils doit être ≥ 1".into()));
    }
    valider_champ(champ)?;
    let table = Table::nouvelle(opt.geometrie);
    let lc = champ.l / 3;
    let vc = lc.pow(4);
    let n_liens = 4 * vc;
    let n_morceaux = n_liens.div_ceil(64);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(opt.fils)
        .build()
        .map_err(|e| Erreur::Config(format!("création du pool de fils impossible : {e}")))?;
    // broadcast garantit une instance de tampon sur chaque fil réel du pool local.
    let par_fil = pool.broadcast(|contexte| {
        let mut tampon = Tampon::nouveau(&table);
        let mut morceaux = Vec::new();
        for numero in (contexte.index()..n_morceaux).step_by(opt.fils) {
            let debut = 64 * numero;
            let fin = (debut + 64).min(n_liens);
            let mut quaternions = Vec::with_capacity(fin - debut);
            let mut stats = StatsTheta::vide();
            for li in debut..fin {
                let (u, mesures) = tampon.lien(
                    champ,
                    &table,
                    coordonnees(li % vc, lc),
                    li / vc,
                    opt.angle_so3,
                );
                quaternions.push(u);
                stats.fusionner(&mesures);
            }
            morceaux.push((numero, quaternions, stats));
        }
        morceaux
    });
    let mut morceaux: Vec<_> = par_fil.into_iter().flatten().collect();
    morceaux.sort_unstable_by_key(|m| m.0);
    let mut q = Vec::with_capacity(4 * n_liens);
    let mut stats = StatsTheta::vide();
    for (_, quaternions, mesures) in morceaux {
        for u in quaternions {
            q.extend_from_slice(&u);
        }
        stats.fusionner(&mesures);
    }
    Ok(SortieBloc {
        grossier: ChampHote { l: lc, q },
        stats,
    })
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::su2::cpu::haar_q;

    pub(in crate::rg) fn proche(a: Q, b: Q, seuil: f64) {
        for (x, y) in a.into_iter().zip(b) {
            assert!(
                (x - y).abs() < seuil,
                "écart {} ≥ {seuil} : {a:?}, {b:?}",
                (x - y).abs()
            );
        }
    }

    pub(in crate::rg) fn froid(l: usize) -> ChampHote {
        ChampHote {
            l,
            q: (0..4 * l.pow(4)).flat_map(|_| IDENTITE).collect(),
        }
    }

    pub(in crate::rg) fn aleatoire(l: usize) -> ChampHote {
        ChampHote {
            l,
            q: (0..4 * l.pow(4))
                .flat_map(|i| haar_q(i as u32, 0, 23, 17, 41))
                .collect(),
        }
    }

    pub(in crate::rg) fn jauges(l: usize) -> Vec<Q> {
        (0..l.pow(4))
            .map(|i| haar_q(i as u32, 1, 29, 43, 13))
            .collect()
    }

    pub(in crate::rg) fn transformer(champ: &ChampHote, g: &[Q]) -> ChampHote {
        let mut resultat = champ.clone();
        for mu in 0..4 {
            for s in 0..champ.v() {
                let x = coordonnees(s, champ.l);
                let t = site(deplacer(x, mu, 1, champ.l), champ.l);
                let u = qmul(qmul(g[s], champ.lien(mu * champ.v() + s)), qdag(g[t]));
                resultat.q[4 * (mu * champ.v() + s)..4 * (mu * champ.v() + s) + 4]
                    .copy_from_slice(&u);
            }
        }
        resultat
    }

    fn options(fils: usize) -> OptionsBloc {
        OptionsBloc {
            geometrie: Geometrie::TroisCentre,
            fils,
            angle_so3: false,
        }
    }

    #[test]
    fn aller_retour_log_exp_philox() {
        for i in 0..1000 {
            let q = haar_q(i, 7, 31, 19, 47);
            let s = log_q(q).0;
            proche(exp_q(s), q, 1e-14);
            let t = log_q(exp_q(s)).0;
            for m in 0..3 {
                assert!((s[m] - t[m]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn limites_log_exp() {
        assert_eq!(log_q(IDENTITE), ([0.0; 3], 0.0, false));
        assert_eq!(exp_q([0.0; 3]), IDENTITE);
        assert_eq!(log_q([-1.0, 0.0, 0.0, 0.0]), ([0.0; 3], PI, true));
        let s = [1e-160, -2e-160, 3e-160];
        assert_eq!(log_q(exp_q(s)).0, s);
        let (v, theta, ambigu) = log_q(exp_q([PI - 5e-7, 0.0, 0.0]));
        assert!(ambigu);
        assert!((theta - (PI - 5e-7)).abs() < 1e-15);
        // La série sous 1e-6 est celle imposée, même près de l'antipode.
        assert!((v[0] - 5e-7).abs() < 1e-15);
        assert!(!log_q(exp_q([PI - 2e-6, 0.0, 0.0])).2);
        for amplitude in [5e-7, 1e-6, 2e-6] {
            let s = [amplitude, 0.0, 0.0];
            assert!((log_q(exp_q(s)).0[0] - amplitude).abs() < 1e-14);
        }
    }

    #[test]
    fn statistiques_et_fusion() {
        let mut a = StatsTheta::vide();
        assert_eq!(a.fraction_hors(), 0.0);
        a.ajouter(0.0, false, false);
        a.ajouter(PI / 2.0, false, false);
        let mut b = StatsTheta::vide();
        b.ajouter(PI, true, false);
        b.ajouter(PI / 3.0, false, true);
        a.fusionner(&b);
        assert_eq!(a.n, 4);
        assert_eq!(a.n_hors, 2);
        assert_eq!(a.n_ambigues, 1);
        assert_eq!(a.fraction_hors(), 0.5);
        assert_eq!(a.theta_max, PI);
        assert_eq!(a.histogramme[0], 1);
        assert_eq!(a.histogramme[32], 1);
        assert_eq!(a.histogramme[42], 1);
        assert_eq!(a.histogramme[63], 1);
        b.ajouter(PI, false, true);
        assert_eq!(b.theta_max, 2.0 * PI);
        assert_eq!(b.histogramme[63], 2);
        let copie = a.clone();
        a.fusionner(&StatsTheta::vide());
        assert_eq!(a, copie);
    }

    #[test]
    fn froid_et_compte_complet() {
        for g in [Geometrie::TroisCentre, Geometrie::TroisCoin] {
            let opt = OptionsBloc {
                geometrie: g,
                ..options(4)
            };
            let sortie = bloquer(&froid(9), &opt).unwrap();
            assert_eq!(sortie.grossier, froid(3));
            assert_eq!(
                sortie.stats.n,
                4 * 81 * Table::nouvelle(g).nb_boucles() as u64
            );
            assert_eq!(sortie.stats.histogramme[0], sortie.stats.n);
            assert_eq!(sortie.stats.theta_max, 0.0);
            assert_eq!(sortie.stats.n_ambigues, 0);
            assert_eq!(sortie.stats.n_hors, 0);
        }
        assert_eq!(bloquer(&froid(3), &options(1)).unwrap().grossier, froid(1));
    }

    #[test]
    fn jauge_pure_l9() {
        let g = jauges(9);
        let champ = transformer(&froid(9), &g);
        let sortie = bloquer(&champ, &options(4)).unwrap();
        for mu in 0..4 {
            for s in 0..81 {
                let c = coordonnees(s, 3).map(|v| 3 * v + 1);
                let cp = deplacer(c, mu, 3, 9);
                proche(
                    sortie.grossier.lien(mu * 81 + s),
                    qmul(g[site(c, 9)], qdag(g[site(cp, 9)])),
                    1e-14,
                );
            }
        }
    }

    #[test]
    fn covariance_l9() {
        let champ = aleatoire(9);
        let g = jauges(9);
        let a = bloquer(&champ, &options(4)).unwrap();
        let b = bloquer(&transformer(&champ, &g), &options(4)).unwrap();
        for mu in 0..4 {
            for s in 0..81 {
                let c = coordonnees(s, 3).map(|v| 3 * v + 1);
                let cp = deplacer(c, mu, 3, 9);
                let attendu = qmul(
                    qmul(g[site(c, 9)], a.grossier.lien(mu * 81 + s)),
                    qdag(g[site(cp, 9)]),
                );
                proche(b.grossier.lien(mu * 81 + s), attendu, 1e-13);
            }
        }
    }

    #[test]
    fn identite_des_octets_selon_les_fils() {
        let champ = aleatoire(9);
        let a = bloquer(&champ, &options(1)).unwrap();
        for fils in [3, 8] {
            let b = bloquer(&champ, &options(fils)).unwrap();
            assert_eq!(a.stats, b.stats);
            assert_eq!(a.stats.theta_max.to_bits(), b.stats.theta_max.to_bits());
            assert!(a
                .grossier
                .q
                .iter()
                .zip(&b.grossier.q)
                .all(|(x, y)| x.to_bits() == y.to_bits()));
        }
        let b = bloquer(
            &champ,
            &OptionsBloc {
                angle_so3: true,
                ..options(4)
            },
        )
        .unwrap();
        assert_eq!(a.grossier, b.grossier);
        assert_eq!(a.stats.n, b.stats.n);
        assert_eq!(a.stats.n_ambigues, b.stats.n_ambigues);
        assert_eq!(2.0 * a.stats.theta_max, b.stats.theta_max);
        assert!(b.stats.n_hors >= a.stats.n_hors);
    }

    #[test]
    fn entrees_invalides() {
        for l in [0, 1, 2, 4] {
            assert!(matches!(
                bloquer(&froid(l), &options(1)),
                Err(Erreur::Config(_))
            ));
        }
        assert!(matches!(
            bloquer(&froid(3), &options(0)),
            Err(Erreur::Config(_))
        ));
        let mut champ = froid(3);
        champ.q.pop();
        assert!(bloquer(&champ, &options(1)).is_err());
        for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut champ = froid(3);
            champ.q[13] = v;
            assert!(matches!(
                bloquer(&champ, &options(1)),
                Err(Erreur::Config(_))
            ));
        }
        assert!(bloquer(
            &ChampHote {
                l: usize::MAX,
                q: Vec::new()
            },
            &options(1)
        )
        .is_err());
    }
}
