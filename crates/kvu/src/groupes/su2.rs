//! Fiche SU(2), générateurs σᵃ/2 et convention quaternionique du crate.

use super::{produit, quotient, Appartenance, Cmat, Flux, Groupe, LogU, Realite};
use crate::heritage::su2_hb;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutantSu2 {
    Aucun,
    GenerateurRetire,
    QrNaif,
}

pub struct Su2 {
    mutant: MutantSu2,
}

impl Default for Su2 {
    fn default() -> Self {
        Self::nouveau()
    }
}

impl Su2 {
    pub fn nouveau() -> Self {
        Self {
            mutant: MutantSu2::Aucun,
        }
    }

    /// Contrôles NÉGATIFS de la batterie G-MUT, jamais en production.
    pub fn avec_mutant(mutant: MutantSu2) -> Self {
        Self { mutant }
    }

    pub fn vers_mat(q: [f64; 4]) -> Cmat {
        Cmat::depuis_entrelace(2, &su2_hb::quat_to_mat(q))
    }

    /// Inverse sur SU(2) ; moyenne des composantes sur une matrice quelconque.
    pub fn vers_quat(m: &Cmat) -> [f64; 4] {
        verifier_dimension(m);
        su2_hb::mat_to_quat(&m.vers_entrelace())
    }
}

fn verifier_dimension(m: &Cmat) {
    assert_eq!(m.n, 2, "SU(2) exige une matrice de dimension 2×2");
    // Vérifie aussi les longueurs du stockage public de Cmat.
    let _ = m.get(0, 0);
}

fn determinant_deux(m: &Cmat) -> (f64, f64) {
    let ad = produit(m.get(0, 0), m.get(1, 1));
    let bc = produit(m.get(0, 1), m.get(1, 0));
    (ad.0 - bc.0, ad.1 - bc.1)
}

fn inverse_deux(m: &Cmat) -> Cmat {
    let det = determinant_deux(m);
    let mut inverse = Cmat::zeros(2);
    inverse.set(0, 0, quotient(m.get(1, 1), det));
    inverse.set(1, 1, quotient(m.get(0, 0), det));
    let b = m.get(0, 1);
    let c = m.get(1, 0);
    inverse.set(0, 1, quotient((-b.0, -b.1), det));
    inverse.set(1, 0, quotient((-c.0, -c.1), det));
    inverse
}

fn racine_principale(z: (f64, f64)) -> (f64, f64) {
    let module = z.0.hypot(z.1);
    if z.0 >= 0.0 {
        let re = (0.5 * module + 0.5 * z.0).sqrt();
        (re, z.1 / (2.0 * re))
    } else {
        let im = (0.5 * module - 0.5 * z.0).sqrt().copysign(z.1);
        (z.1 / (2.0 * im), im)
    }
}

fn normaliser_colonne(m: &mut Cmat, j: usize) {
    let a = m.get(0, j);
    let b = m.get(1, j);
    let norme = a.0.hypot(a.1).hypot(b.0.hypot(b.1));
    assert!(norme > 0.0, "Colonne gaussienne nulle pendant Gram–Schmidt");
    m.set(0, j, (a.0 / norme, a.1 / norme));
    m.set(1, j, (b.0 / norme, b.1 / norme));
}

/// Un tirage du Flux par matrice ; k distingue les quatre entrées gaussiennes.
fn haar_qr_naif(rng: &mut Flux) -> Cmat {
    let (i, slot, k0, k1, etiquette) = rng.suivant();
    let mut m = Cmat::zeros(2);
    for k in 0..4 {
        let mots = su2_hb::philox4x32_10([i, slot, k, etiquette], k0, k1);
        let u = su2_hb::u01(mots[0], mots[1]);
        let v = su2_hb::u01(mots[2], mots[3]);
        let rayon = (-2.0 * u.ln()).sqrt();
        let (sin, cos) = (2.0 * PI * v).sin_cos();
        m.set(k as usize / 2, k as usize % 2, (rayon * cos, rayon * sin));
    }
    normaliser_colonne(&mut m, 0);
    // Réorthogonalisation : réduit l'erreur d'annulation pour les colonnes proches.
    // Les normes divisées sont réelles positives : aucune correction de phase de det.
    for _ in 0..2 {
        let mut recouvrement = (0.0, 0.0);
        for i in 0..2 {
            let a = m.get(i, 0);
            let p = produit((a.0, -a.1), m.get(i, 1));
            recouvrement.0 += p.0;
            recouvrement.1 += p.1;
        }
        for i in 0..2 {
            let b = m.get(i, 1);
            let p = produit(m.get(i, 0), recouvrement);
            m.set(i, 1, (b.0 - p.0, b.1 - p.1));
        }
    }
    normaliser_colonne(&mut m, 1);
    m
}

impl Groupe for Su2 {
    fn nom(&self) -> &'static str {
        "su2"
    }

    fn d_f(&self) -> usize {
        2
    }

    fn d_adj(&self) -> usize {
        3
    }

    fn realite(&self) -> Realite {
        Realite::Pseudoreel
    }

    fn h_vee(&self) -> usize {
        2
    }

    fn generateurs(&self) -> Vec<Cmat> {
        let mut t = vec![
            Cmat::depuis_entrelace(2, &[0., 0., 0.5, 0., 0.5, 0., 0., 0.]),
            Cmat::depuis_entrelace(2, &[0., 0., 0., -0.5, 0., 0.5, 0., 0.]),
        ];
        if self.mutant != MutantSu2::GenerateurRetire {
            t.push(Cmat::depuis_entrelace(
                2,
                &[0.5, 0., 0., 0., 0., 0., -0.5, 0.],
            ));
        }
        t
    }

    fn appartient(&self, u: &Cmat) -> Appartenance {
        verifier_dimension(u);
        let det = u.det();
        Appartenance {
            unitarite: u.dag().mul(u).max_abs_diff(&Cmat::identite(2)),
            det: (det.0 - 1.0).hypot(det.1),
            structure: 0.0,
        }
    }

    fn haar(&self, rng: &mut Flux) -> Cmat {
        if self.mutant == MutantSu2::QrNaif {
            return haar_qr_naif(rng);
        }
        let (i, slot, k0, k1, etiquette) = rng.suivant();
        Self::vers_mat(crate::su2::cpu::haar_q(i, slot, etiquette, k0, k1))
    }

    fn projeter(&self, m: &Cmat) -> Cmat {
        verifier_dimension(m);
        let det = determinant_deux(m);
        if det.0.hypot(det.1) < 1e-300 {
            return Cmat::identite(2);
        }
        let mut u = m.clone();
        for _ in 0..60 {
            let suivante = u.add(&inverse_deux(&u.dag())).echelle(0.5);
            let difference = suivante.max_abs_diff(&u);
            u = suivante;
            if difference < 1e-16 {
                break;
            }
        }
        let racine = racine_principale(determinant_deux(&u));
        u.echelle_c(quotient((1.0, 0.0), racine))
    }

    fn exp(&self, a: &[f64]) -> Cmat {
        assert_eq!(
            a.len(),
            3,
            "SU(2) exige trois coordonnées pour l'exponentielle"
        );
        let norme = a[0].hypot(a[1]).hypot(a[2]);
        if norme < 1e-300 {
            return Cmat::identite(2);
        }
        let theta = 0.5 * norme;
        let facteur = if norme < 1e-4 {
            let x2 = theta * theta;
            0.5 * (1.0 - x2 / 6.0 + x2 * x2 / 120.0)
        } else {
            theta.sin() / norme
        };
        Self::vers_mat([theta.cos(), facteur * a[0], facteur * a[1], facteur * a[2]])
    }

    fn log(&self, u: &Cmat) -> LogU {
        let q = Self::vers_quat(u);
        let norme = q[1].hypot(q[2]).hypot(q[3]);
        let theta = norme.atan2(q[0]);
        let a = if norme < 1e-300 {
            vec![0.0; 3]
        } else {
            q[1..].iter().map(|&x| 2.0 * theta * (x / norme)).collect()
        };
        LogU {
            a,
            theta,
            ambigue: PI - theta < 1e-6,
            hors_domaine: theta > PI / 2.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commun::Graine;
    use crate::groupes::ETIQUETTE_GROUPE;

    fn flux() -> Flux {
        Flux::nouveau(Graine(0x1234_ABCD_5678_EF90), ETIQUETTE_GROUPE)
    }

    fn uniformes(rng: &mut Flux) -> [f64; 4] {
        let (i, slot, k0, k1, etiquette) = rng.suivant();
        std::array::from_fn(|k| {
            let mots = su2_hb::philox4x32_10([i, slot, k as u32, etiquette], k0, k1);
            su2_hb::u01(mots[0], mots[1])
        })
    }

    fn bits(m: &Cmat) -> Vec<u64> {
        m.vers_entrelace().iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn fiche_et_generateurs() {
        let g = Su2::nouveau();
        assert_eq!((g.nom(), g.d_f(), g.d_adj(), g.h_vee()), ("su2", 2, 3, 2));
        assert_eq!(g.realite(), Realite::Pseudoreel);
        assert!((g.t_f() - 0.5).abs() <= 1e-15);
        let t = g.generateurs();
        assert_eq!(t.len(), 3);
        for ta in &t {
            assert!(ta.max_abs_diff(&ta.dag()) <= 1e-15);
            let trace = ta.trace();
            assert!(trace.0.hypot(trace.1) <= 1e-15);
        }
        for (a, ta) in t.iter().enumerate() {
            for (b, tb) in t.iter().enumerate() {
                let trace = ta.mul(tb).trace();
                assert!((trace.0 - if a == b { 0.5 } else { 0.0 }).abs() <= 1e-15);
                assert!(trace.1.abs() <= 1e-15);
            }
        }
    }

    #[test]
    fn constantes_de_structure_et_contraction() {
        let g = Su2::nouveau();
        let f = g.f_abc();
        assert_eq!(f.len(), 27);
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    let epsilon = if a == b || b == c || a == c {
                        0.0
                    } else if (a + 1) % 3 == b {
                        1.0
                    } else {
                        -1.0
                    };
                    assert!((f[(a * 3 + b) * 3 + c] - epsilon).abs() <= 1e-15);
                }
                let mut contraction = 0.0;
                for c in 0..3 {
                    for d in 0..3 {
                        contraction += f[(a * 3 + c) * 3 + d] * f[(b * 3 + c) * 3 + d];
                    }
                }
                assert!((contraction - if a == b { 2.0 } else { 0.0 }).abs() <= 1e-14);
            }
        }
    }

    #[test]
    fn conversions_et_reference_philox() {
        let q = [0.5, -0.5, 0.5, -0.5];
        let m = Su2::vers_mat(q);
        assert_eq!(
            m.vers_entrelace(),
            vec![0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5]
        );
        assert_eq!(Su2::vers_quat(&m), q);
        let mut rng = flux();
        let mut reference = flux();
        for _ in 0..100 {
            let (i, slot, k0, k1, etiquette) = reference.suivant();
            let attendu = Su2::vers_mat(crate::su2::cpu::haar_q(i, slot, etiquette, k0, k1));
            assert_eq!(bits(&Su2::nouveau().haar(&mut rng)), bits(&attendu));
        }
        assert_eq!(rng.tirages(), 100);
    }

    #[test]
    fn haar_unitarite_et_determinant_cent_mille() {
        let g = Su2::nouveau();
        let mut rng = flux();
        let (mut max_unite, mut max_det) = (0.0_f64, 0.0_f64);
        for _ in 0..100_000 {
            let appartenance = g.appartient(&g.haar(&mut rng));
            max_unite = max_unite.max(appartenance.unitarite);
            max_det = max_det.max(appartenance.det);
            assert_eq!(appartenance.structure, 0.0);
        }
        assert!(
            max_unite <= 1e-14,
            "Défaut maximal d'unitarité : {max_unite}"
        );
        assert!(
            max_det <= 1e-14,
            "Défaut maximal de déterminant : {max_det}"
        );
    }

    #[test]
    fn haar_moments_deux_cent_mille() {
        let g = Su2::nouveau();
        let mut rng = flux();
        let n = 200_000;
        let mut sommes = [0.0; 3];
        for _ in 0..n {
            let u = g.haar(&mut rng);
            let trace = u.trace();
            let carre_trace = produit(trace, trace);
            let trace_carre = u.mul(&u).trace();
            for (k, z) in [trace, carre_trace, trace_carre].iter().enumerate() {
                sommes[k] += z.0;
                assert!(z.1.abs() <= 1e-15);
            }
        }
        // Chacune de ces trois observables réelles a une variance de 1 sous Haar SU(2).
        let quatre_sigma = 4.0 / (n as f64).sqrt();
        for (somme, cible) in sommes.iter().zip([0.0, 1.0, -1.0]) {
            let moyenne = somme / n as f64;
            assert!(
                (moyenne - cible).abs() <= quatre_sigma,
                "Moment {moyenne}, cible {cible}, seuil {quatre_sigma}"
            );
        }
    }

    #[test]
    fn reproductibilite_et_etiquettes() {
        for mutant in [
            MutantSu2::Aucun,
            MutantSu2::GenerateurRetire,
            MutantSu2::QrNaif,
        ] {
            let g = Su2::avec_mutant(mutant);
            let (mut a, mut b) = (flux(), flux());
            let mut autre = Flux::nouveau(Graine(0x1234_ABCD_5678_EF90), ETIQUETTE_GROUPE + 1);
            for _ in 0..100 {
                let u = g.haar(&mut a);
                assert_eq!(bits(&u), bits(&g.haar(&mut b)));
                assert_ne!(bits(&u), bits(&g.haar(&mut autre)));
            }
        }
    }

    #[test]
    fn mutant_generateur_retire() {
        let mutant = Su2::avec_mutant(MutantSu2::GenerateurRetire);
        let g = Su2::nouveau();
        assert_eq!(mutant.generateurs(), g.generateurs()[..2]);
        assert_eq!(mutant.d_adj(), 3);
        assert_eq!(mutant.t_f(), 0.5);
        assert_eq!(mutant.f_abc(), vec![0.0; 8]);
        let (mut a, mut b) = (flux(), flux());
        assert_eq!(bits(&mutant.haar(&mut a)), bits(&g.haar(&mut b)));
        let coordonnees = [0.3, -0.5, 0.7];
        assert_eq!(mutant.exp(&coordonnees), g.exp(&coordonnees));
    }

    #[test]
    fn mutant_qr_unitaire_mais_pas_su2() {
        let g = Su2::avec_mutant(MutantSu2::QrNaif);
        let mut rng = flux();
        let mut det_variable = false;
        let n = 20_000;
        let mut sommes = [0.0; 2];
        let mut carres = [0.0; 2];
        for i in 0..n {
            let u = g.haar(&mut rng);
            let appartenance = g.appartient(&u);
            assert!(
                appartenance.unitarite <= 1e-15,
                "Défaut QR : {}",
                appartenance.unitarite
            );
            let det = u.det();
            assert!((det.0.hypot(det.1) - 1.0).abs() <= 1e-14);
            if i < 100 && appartenance.det > 0.1 {
                det_variable = true;
            }
            let z = u.mul(&u).trace();
            for (k, valeur) in [z.0, z.1].into_iter().enumerate() {
                sommes[k] += valeur;
                carres[k] += valeur * valeur;
            }
        }
        assert!(det_variable, "Le mutant QR doit sortir de SU(2)");
        for k in 0..2 {
            let moyenne = sommes[k] / n as f64;
            let variance = (carres[k] - sommes[k] * moyenne) / (n - 1) as f64;
            let quatre_sigma = 4.0 * (variance / n as f64).sqrt();
            assert!(
                moyenne.abs() <= quatre_sigma,
                "Moment QR {moyenne}, seuil {quatre_sigma}"
            );
            if k == 0 {
                assert!((moyenne + 1.0).abs() > quatre_sigma);
            }
        }
    }

    #[test]
    fn exp_log_dix_mille_vecteurs() {
        let g = Su2::nouveau();
        let mut rng = flux();
        for _ in 0..10_000 {
            let u = uniformes(&mut rng);
            let direction = [2.0 * u[0] - 1.0, 2.0 * u[1] - 1.0, 2.0 * u[2] - 1.0];
            let norme = direction[0].hypot(direction[1]).hypot(direction[2]);
            let rayon = (2.0 * PI - 0.1) * u[3];
            let a = direction.map(|x| x * rayon / norme);
            let m = g.exp(&a);
            let log = g.log(&m);
            for (&x, &y) in log.a.iter().zip(&a) {
                assert!((x - y).abs() <= 1e-12, "Aller-retour : {x}, {y}");
            }
            assert!(g.exp(&log.a).max_abs_diff(&m) <= 1e-14);
            let oppose = a.map(|x| -x);
            assert!(m.mul(&g.exp(&oppose)).max_abs_diff(&Cmat::identite(2)) <= 1e-15);
        }
    }

    #[test]
    fn exp_log_limites_et_petits_angles() {
        let g = Su2::nouveau();
        let identite = Cmat::identite(2);
        let zero = g.log(&identite);
        assert_eq!(zero.a, vec![0.0; 3]);
        assert_eq!(zero.theta, 0.0);
        assert!(!zero.ambigue && !zero.hors_domaine);
        let moins_un = g.log(&identite.echelle(-1.));
        assert_eq!(moins_un.theta, PI);
        assert!(moins_un.ambigue && moins_un.hors_domaine);
        assert_eq!(moins_un.a, vec![0.0; 3]);
        assert_eq!(g.exp(&[0.0; 3]), identite);
        assert_eq!(g.exp(&[1e-301, 0.0, 0.0]), identite);
        for angle in [1e-299, 1e-10, 1e-6, 1e-5] {
            let a = [angle, -angle, angle];
            let log = g.log(&g.exp(&a));
            for (&x, &y) in log.a.iter().zip(&a) {
                assert!((x - y).abs() <= 1e-14 * angle);
            }
        }
        for (theta, hors_domaine) in [(PI / 2.0 - 1e-8, false), (PI / 2.0 + 1e-8, true)] {
            let log = g.log(&g.exp(&[2.0 * theta, 0.0, 0.0]));
            assert_eq!(log.hors_domaine, hors_domaine);
            assert!(!log.ambigue);
            assert!((log.theta - theta).abs() <= 1e-15);
        }
        for (ecart, ambigue) in [(0.5e-6, true), (2e-6, false)] {
            assert_eq!(
                g.log(&g.exp(&[2.0 * (PI - ecart), 0.0, 0.0])).ambigue,
                ambigue
            );
        }
    }

    #[test]
    fn projection_idempotence_et_bruit() {
        let g = Su2::nouveau();
        let mut rng = flux();
        for _ in 0..1_000 {
            let u = g.haar(&mut rng);
            let p = g.projeter(&u);
            assert!(p.max_abs_diff(&u) <= 1e-15);
            assert!(g.projeter(&p).max_abs_diff(&p) <= 1e-14);
            let mut bruit = Cmat::zeros(2);
            for i in 0..2 {
                let valeurs = uniformes(&mut rng);
                for j in 0..2 {
                    bruit.set(
                        i,
                        j,
                        (2.0 * valeurs[2 * j] - 1.0, 2.0 * valeurs[2 * j + 1] - 1.0),
                    );
                }
            }
            let proche = g.projeter(&u.add(&bruit.echelle(1e-3)));
            assert!(proche.max_abs_diff(&u) < 5e-3);
            let appartenance = g.appartient(&proche);
            assert!(appartenance.unitarite <= 1e-14 && appartenance.det <= 1e-14);
        }
        assert_eq!(g.projeter(&Cmat::zeros(2)), Cmat::identite(2));
        let singuliere = Cmat::depuis_entrelace(2, &[1., 1., 2., 0., 2., 2., 4., 0.]);
        assert_eq!(g.projeter(&singuliere), Cmat::identite(2));
    }

    #[test]
    fn projection_determinant_complexe_et_racine_principale() {
        let g = Su2::nouveau();
        for m in [
            Cmat::depuis_entrelace(2, &[1., 2., 3., -1., -2., 0.5, 4., 3.]),
            Cmat::depuis_entrelace(2, &[-2., 0., 0., 0., 0., 0., 3., 0.]),
            Cmat::identite(2).echelle_c((0., 2.)),
            Cmat::identite(2).echelle_c((1., -3.)),
        ] {
            let p = g.projeter(&m);
            let appartenance = g.appartient(&p);
            assert!(
                appartenance.det <= 1e-14,
                "Déterminant projeté : {}",
                appartenance.det
            );
            assert!(appartenance.unitarite <= 1e-14);
        }
        // Sur une diagonale, le facteur polaire et la racine principale se calculent à la main.
        let m = Cmat::depuis_entrelace(2, &[0., 2., 0., 0., 0., 0., 3., 0.]);
        let r = 0.5_f64.sqrt();
        let attendu = Cmat::depuis_entrelace(2, &[r, r, 0., 0., 0., 0., r, -r]);
        assert!(g.projeter(&m).max_abs_diff(&attendu) <= 1e-15);
    }
}
