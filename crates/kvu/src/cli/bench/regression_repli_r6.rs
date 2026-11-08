
    use super::*;

    #[test]
    fn contre_exemple_historique_du_repli_conserve() {
        let m = estimer_mur(&[0.01, 0.01, 0.01], &[0.04, 0.035], 10, 40, 0.002);
        assert_eq!(m.estimation, "borne_haute_t_sur_n");
        assert!(1.0 / m.s_par_pas > TOLERANCE_MUR_NOYAUX / 0.002);
        // Cette valeur n'est plus publiée : le banc réel utilise ProfilRun.
    }
