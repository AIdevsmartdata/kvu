//! `thermo mesure --beta-s` : couplages d'espace et de temps séparés (sprint 22/11/2026). Tests sans carte sur les tables.
use kvu::commun::{Depart, Graine, Precision};
use kvu::su2::torsion::Torsion;
use kvu::su2::une_boite::{poids_plans_st, tables_boite_fixe, tables_boite_gen, ParamBoite};
use kvu::su2::ParamChaine;

fn pb(cotes: [usize; 4], tors: &str) -> ParamBoite {
    ParamBoite {
        cotes,
        torsion: Torsion::analyser(tors).unwrap(),
        poids: 1.0,
        nor: 2,
        chaines: vec![ParamChaine { beta: 2.0, graine: Graine(1), depart: Depart::Froid }],
        precision: Precision::F64,
        impairs: false,
        aniso: None,
        couture: None,
    }
}

/// Sans l'option : exactement `tables_boite_fixe` (coefficients et masque, au bit).
#[test]
fn st_none_est_l_historique() {
    for tors in ["aucune", "01"] {
        let p = pb([4, 4, 4, 4], tors);
        let (_, a, fa) = tables_boite_fixe(&p, None).unwrap();
        let (_, b, fb) = tables_boite_gen(&p, None, None).unwrap();
        assert_eq!(fa, fb);
        assert_eq!(a.map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>()), b.map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>()));
    }
}

/// Poids : 1 pour 12, 13, 23 ; wt pour 01, 02, 03 ; masque de feuille = plans 0i seulement ; wt = 0 admis.
#[test]
fn st_poids_et_masque() {
    let p = pb([4, 6, 4, 8], "aucune");
    for wt in [0.0, 0.095, 1.0] {
        let (t, coef, fe) = tables_boite_gen(&p, None, Some((1.0, wt))).unwrap();
        let (v, c) = (t.v, coef.unwrap());
        let w = poids_plans_st(1.0, wt);
        assert_eq!(w, [wt, wt, wt, 1.0, 1.0, 1.0]);
        for i in 0..6 {
            assert!(c[i * v..(i + 1) * v].iter().all(|&x| x == w[i]));
            assert!(fe[i * v..(i + 1) * v].iter().all(|&x| x == u8::from(i < 3)));
        }
    }
}

/// Refus propre avec torsion (--torsion, --torsion-fixe) ou --xi0 ; poids négatif refusé.
#[test]
fn st_refus() {
    assert!(tables_boite_gen(&pb([4, 4, 4, 4], "01"), None, Some((1.0, 0.5))).is_err());
    let f = Torsion::analyser("01").unwrap();
    assert!(tables_boite_gen(&pb([4, 4, 4, 4], "aucune"), Some(&f), Some((1.0, 0.5))).is_err());
    let mut p = pb([4, 4, 4, 4], "aucune");
    p.aniso = Some((1.5, 0));
    assert!(tables_boite_gen(&p, None, Some((1.0, 0.5))).is_err());
    assert!(tables_boite_gen(&pb([4, 4, 4, 4], "aucune"), None, Some((1.0, -0.1))).is_err());
}
