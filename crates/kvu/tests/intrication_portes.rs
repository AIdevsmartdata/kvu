//! Portes du module intrication (tailles réduites ; les versions longues : `kvu intrication portes`).
use kvu::intrication::portes;
use kvu::intrication::{GeoRep, Mutant, Region, Reseau};

#[test]
fn a_region_vide_bit_a_bit_cpusu2() {
    let a = portes::porte_a(4, 2.3, 3, 2, [0x1234_5678_9ABC, 0x0F0F_1111_2222]).unwrap();
    assert_eq!(a.liens_differents, 0, "{a:?}");
    // plaquette moyenne de chaque réplique (liste du module) contre la mesure de CpuSu2 (autre ordre de produit et de somme :
    // égalité à l'arrondi près, pas au bit) ; remplace une comparaison d'une valeur à elle-même (relecture, objection 5)
    for r in 0..2 {
        assert!(
            (a.plaq_replique[r] - a.plaq_cpusu2[r]).abs() < 1e-12,
            "réplique {r} : {a:?}"
        );
    }
    assert!(
        (a.plaq_replique[0] - a.plaq_replique[1]).abs() > 1e-6,
        "répliques indiscernables : {a:?}"
    );
}

#[test]
fn b_region_pleine_egale_reseau_2t() {
    for (l, t) in [(2, 2), (4, 2), (4, 4)] {
        let b = portes::porte_b(l, t, 77, Mutant::Aucun).unwrap();
        assert_eq!(
            (
                b.agrafes_differentes,
                b.plaquettes_differentes,
                b.poids_non_unitaires
            ),
            (0, 0, 0),
            "{b:?}"
        );
        assert_eq!(b.ecart_action, 0.0);
    }
}

#[test]
fn f_mutant_agrafe_coupure_rougit_b_et_c() {
    let b = portes::porte_b(4, 4, 77, Mutant::AgrafeCoupure).unwrap();
    assert!(
        b.agrafes_differentes > 0 && b.plaquettes_differentes > 0,
        "{b:?}"
    );
    // largeur 2 sur L = 4 : tous les sites de la tranche sont des sommets de branchement (jauge identifiée), le mutant y
    // est invisible pour (c) ; largeur 3 a des sites intérieurs à A.
    let c = portes::porte_c(4, 4, 3, 5, Mutant::AgrafeCoupure, true).unwrap();
    assert!(c.ecart_max_plaquette > 1e-3, "{c:?}");
}

#[test]
fn c_invariance_de_jauge() {
    for larg in [0, 1, 2, 3, 4] {
        let c = portes::porte_c(4, 4, larg, 9 + larg as u64, Mutant::Aucun, true).unwrap();
        assert!(
            c.ecart_max_plaquette < 1e-12 && c.ecart_action_relatif < 1e-12,
            "{c:?}"
        );
    }
    // sans identification aux sommets de branchement, la jauge n'est PAS une symétrie (la porte voit l'écart)
    let c = portes::porte_c(4, 4, 2, 5, Mutant::Aucun, false).unwrap();
    assert!(
        c.sommets_branchement > 0 && c.ecart_max_plaquette > 1e-3,
        "{c:?}"
    );
}

#[test]
fn d1_agrafe_egale_variation_action() {
    for (larg, alpha) in [(0usize, 0.5f64), (1, 0.37), (2, 0.0), (2, 1.0), (3, 0.81)] {
        let d = portes::porte_d1(4, 4, larg, alpha, 3, Mutant::Aucun).unwrap();
        assert!(d.ecart_max < 1e-12, "{d:?}");
    }
    let d = portes::porte_d1(4, 4, 1, 0.37, 4, Mutant::PoidsIgnores).unwrap();
    assert!(d.ecart_max > 1e-3, "{d:?}");
}

#[test]
fn d2_classes_independantes_et_mutant() {
    let g = GeoRep::nouveau(4, 4, 2).unwrap();
    let r = Reseau::nouveau(
        g,
        Region::tranche(&g, 1),
        Region::tranche(&g, 2),
        &[1, 2],
        Mutant::Aucun,
    )
    .unwrap();
    assert_eq!(portes::porte_classes(&r), 0);
    let gi = GeoRep::nouveau_sans_controle(4, 3, 2);
    let ri = Reseau::nouveau(
        gi,
        Region::tranche(&gi, 1),
        Region::tranche(&gi, 2),
        &[1, 2],
        Mutant::Aucun,
    )
    .unwrap();
    assert!(portes::porte_classes(&ri) > 0);
}

#[test]
fn nombre_de_plaquettes_independant_de_la_largeur() {
    // Σ_p w_p = 6V pour toute région (le terme −β N_p s'élimine dans S₂(l+1) − S₂(l))
    let g = GeoRep::nouveau(4, 4, 2).unwrap();
    for larg in 0..=4 {
        let r = Reseau::pur(g, Region::tranche(&g, larg), &[1, 2], Mutant::Aucun).unwrap();
        let s: f64 = r.plaqs.iter().map(|p| p.w0).sum();
        assert_eq!(s, (6 * g.v) as f64, "largeur {larg}");
    }
}

#[test]
fn e_calcul_a_la_main_couplage_fort() {
    // l ≥ 1 : premier ordre de l'intégrande 2β²L²(1 − 2α), intégrale nulle ; l = 0 → 1 : S₂(1) = β²L²/2 au premier ordre
    let beta: f64 = 0.3;
    for alpha in [0.0, 0.3, 1.0] {
        let (_, o1) = portes::prediction_couplage_fort(4, 4, 1, 2, beta, alpha).unwrap();
        assert!(
            (o1 - 2.0 * beta * beta * 16.0 * (1.0 - 2.0 * alpha)).abs() < 1e-12,
            "{o1}"
        );
    }
    let (_, o1) = portes::prediction_df_couplage_fort(4, 4, 0, 1, beta).unwrap();
    assert!((o1 - beta * beta * 16.0 / 2.0).abs() < 1e-12, "{o1}");
    let (_, o1) = portes::prediction_df_couplage_fort(4, 4, 1, 2, beta).unwrap();
    assert!(o1.abs() < 1e-12, "{o1}");
}

#[test]
fn cli_contrat_et_manifeste() {
    use kvu::intrication::cli;
    // contrat des modules kvu : une commande « intrication », sous-commandes analysables sans Parser
    let cmds = cli::commandes();
    assert_eq!(cmds.len(), 1);
    assert_eq!(cmds[0].get_name(), "intrication");
    let rep = std::env::temp_dir().join(format!("kvu_intrication_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&rep);
    let argv = [
        "intrication",
        "balayage",
        "--l",
        "2",
        "--t",
        "2",
        "--l0",
        "0",
        "--l1",
        "1",
        "--beta",
        "1.0",
        "--nalpha",
        "3",
        "--therm",
        "5",
        "--mes",
        "20",
        "--blocs",
        "4",
        "--sortie",
        rep.to_str().unwrap(),
    ];
    let m = cli::commande().try_get_matches_from(argv).unwrap();
    assert_eq!(cli::executer("intrication", &m).unwrap(), 0);
    let man: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(rep.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(man["geometrie"]["dir_temps"], 0, "{man}");
    assert_eq!(man["geometrie"]["cotes"], serde_json::json!([2, 2, 2, 2]));
    // bloc canonique du socle geom : schéma, recollement de largeur 0 (A₀) et 1 (A₁)
    assert_eq!(man["geometrie"]["schema"], "kvu-geometrie/1", "{man}");
    assert_eq!(man["geometrie"]["chemin"], "C2");
    assert_eq!(
        man["geometrie"]["bords"][0]["region"]["largeur"], 0,
        "{man}"
    );
    assert_eq!(
        man["geometrie_alpha1"]["bords"][0]["region"]["largeur"], 1,
        "{man}"
    );
    assert!(rep.join("resultats.jsonl").exists());
    // un répertoire existant est refusé (run neuf)
    assert!(cli::executer("intrication", &m).is_err());
    let _ = std::fs::remove_dir_all(&rep);
}

#[test]
fn geom_depuis_spec_egale_nouveau_et_refus() {
    use kvu::geom::GeoSpec;
    use kvu::intrication::{cli, controler_spec};
    // même réseau, octet pour octet, par la spécification du socle et par la construction directe
    let g = GeoRep::nouveau(4, 6, 2).unwrap();
    let a = Reseau::nouveau(
        g,
        Region::tranche(&g, 1),
        Region::tranche(&g, 2),
        &[3, 4],
        Mutant::Aucun,
    )
    .unwrap();
    let b = Reseau::depuis_spec(
        &cli::spec(4, 6, 1),
        &cli::spec(4, 6, 2),
        &[3, 4],
        Mutant::Aucun,
    )
    .unwrap();
    assert_eq!(a.plaqs, b.plaqs);
    assert_eq!(a.cases, b.cases);
    assert_eq!(a.diff, b.diff);
    // axe 2 : même structure que l'axe 1 à une rotation près (même nombre de plaquettes qui changent)
    let c = Reseau::depuis_spec(
        &GeoSpec::repliques(4, 6, 2, 2, 1),
        &GeoSpec::repliques(4, 6, 2, 2, 2),
        &[3, 4],
        Mutant::Aucun,
    )
    .unwrap();
    assert_eq!(c.diff.len(), a.diff.len());
    // refus : périodique (C0), côtés spatiaux inégaux, côté impair, T impair, dir_temps ≠ 0, largeur > côté
    assert!(controler_spec(&GeoSpec::cubique(4, 0)).is_err());
    let mut s = cli::spec(4, 6, 1);
    s.cotes = [6, 4, 4, 6];
    assert!(controler_spec(&s).is_err());
    assert!(controler_spec(&cli::spec(5, 6, 1)).is_err());
    assert!(controler_spec(&cli::spec(4, 5, 1)).is_err());
    let mut s = cli::spec(4, 6, 1);
    s.dir_temps = 3;
    assert!(controler_spec(&s).is_err());
    assert!(controler_spec(&cli::spec(4, 6, 5)).is_err());
    // deux géométries incompatibles (côtés différents)
    assert!(Reseau::depuis_spec(
        &cli::spec(4, 6, 1),
        &cli::spec(6, 6, 2),
        &[3, 4],
        Mutant::Aucun
    )
    .is_err());
    // s ≠ 2 répliques : refusé (aucune porte ne couvre s = 3 ; relecture, objection 7)
    assert!(controler_spec(&GeoSpec::repliques(4, 6, 3, 1, 1)).is_err());
    assert!(Reseau::depuis_spec(
        &GeoSpec::repliques(4, 6, 3, 1, 1),
        &GeoSpec::repliques(4, 6, 3, 1, 2),
        &[3, 4, 5],
        Mutant::Aucun
    )
    .is_err());
    // A vide et A plein restent acceptés (portes a et b)
    assert!(
        controler_spec(&cli::spec(4, 6, 0)).is_ok() && controler_spec(&cli::spec(4, 6, 4)).is_ok()
    );
}

#[test]
fn balayage_et_doublement_construisent_depuis_la_spec_et_refusent() {
    use kvu::intrication::cli;
    use kvu::intrication::portes::ParamsChaine;
    // relecture, objection 8 : la géométrie du calcul est celle de la spécification (Reseau::depuis_spec), donc contrôlée
    let pc = ParamsChaine {
        therm: 0,
        mes: 2,
        sep: 1,
        n_or: 0,
        blocs: 2,
    };
    assert!(
        portes::balayage(4, 4, 1, 5, 0.3, 3, pc, 1).is_err(),
        "largeur 5 > L = 4 (avant : « tout A » sans erreur)"
    );
    assert!(portes::balayage(4, 4, 1, 2, 0.3, 3, pc, 1).is_ok());
    assert!(
        portes::doublement_specs(&cli::spec(2, 2, 0), &cli::spec(2, 2, 1), 1.0, 3, 3, pc, 1)
            .is_err(),
        "A₁ ≠ tranche entière"
    );
    let b =
        portes::balayage_specs(&cli::spec(2, 2, 0), &cli::spec(2, 2, 1), 1.0, 3, pc, 1).unwrap();
    assert_eq!((b.l, b.t, b.largeur0, b.largeur1), (2, 2, 0, 1));
    // objections 1 et 3 côté bibliothèque : erreur rendue, jamais de panique
    assert!(portes::balayage(2, 2, 0, 1, 1.0, 4, pc, 1).is_err());
    assert!(portes::balayage(2, 2, 0, 1, 0.0, 3, pc, 1).is_err());
    assert!(portes::doublement(2, 2, 1.0, 3, 1, pc, 1).is_err());
}

#[test]
fn debit_refuse_debordement_et_zero_pas() {
    use kvu::intrication::cli;
    // relecture, objection 9 : L³ = 2⁶⁶ débordait (réseau vide, sortie incohérente) ; --pas 0 donnait 0/0
    assert!(GeoRep::nouveau(1 << 22, 2, 2).is_err());
    assert_eq!(cli::debit(1 << 22, 2, 1, 0).unwrap_err().code_sortie(), 2);
    assert_eq!(cli::debit(4, 2, 0, 0).unwrap_err().code_sortie(), 2);
    let v = cli::debit(2, 2, 1, 0).unwrap();
    assert!(v["secondes_par_pas"].as_f64().unwrap().is_finite(), "{v}");
}

#[test]
fn manifeste_clos_en_echec_sur_erreur_ou_panique() {
    use kvu::commun::Erreur;
    use kvu::intrication::cli::{self, Run};
    // relecture, objection 4 : une erreur (ou une panique) après l'ouverture du manifeste le clôt en échec, jamais en_cours
    let base = std::env::temp_dir().join(format!("kvu_intrication_echec_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    let lire = |rep: &std::path::Path| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(rep.join("manifest.json")).unwrap()).unwrap()
    };
    let ouvrir = |nom: &str| {
        let rep = base.join(nom);
        let run = Run::ouvrir(
            &rep,
            "essai",
            &[("geometrie".into(), cli::spec(2, 2, 0))],
            serde_json::json!({"essai": nom}),
        )
        .unwrap();
        assert_eq!(lire(&rep)["statut"], "en_cours");
        (rep, run)
    };
    let (rep, run) = ouvrir("erreur");
    let e = cli::conduire(run, |_| Err(Erreur::Config("échec volontaire".into()))).unwrap_err();
    assert_eq!(e.code_sortie(), 2);
    let m = lire(&rep);
    assert_eq!(
        (m["statut"].as_str(), m["rc"].as_i64()),
        (Some("echec"), Some(2)),
        "{m}"
    );
    assert!(
        m["raison_echec"]
            .as_str()
            .unwrap()
            .contains("échec volontaire"),
        "{m}"
    );
    assert!(!rep.join("COMPLETE").exists());
    assert_eq!(m["parametres"]["statut"], "experimental", "{m}");
    let (rep, run) = ouvrir("panique");
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cli::conduire(run, |_| panic!("panique volontaire de l'essai"))
    }));
    assert!(r.is_err(), "la panique doit reprendre après la clôture");
    assert_eq!(lire(&rep)["statut"], "echec");
    assert!(!rep.join("COMPLETE").exists());
    // témoin : un run sans erreur se termine (COMPLETE en dernier)
    let (rep, run) = ouvrir("ok");
    assert_eq!(
        cli::conduire(run, |r| r.ligne(&serde_json::json!({"x": 1})).map(|_| 0)).unwrap(),
        0
    );
    assert_eq!(lire(&rep)["statut"], "termine");
    assert!(rep.join("COMPLETE").exists());
    let _ = std::fs::remove_dir_all(&base);
}

/// Action recollée réécrite DIRECTEMENT depuis la spécification (note T14 §1, rappelée en tête de `intrication/mod.rs`), sans
/// la table de plaquettes, les fusions ni les cases du module : coordonnées (r, t, x1, x2, x3) et voisins par modulo, champ lu
/// dans la disposition documentée (site = r·V₁ + t·L³ + (x1·L + x2)·L + x3, lien = μ·V + site). Les deux demi-boucles de la
/// couture sont sommées séparément (½ + ½), sans fusion : la somme ne dépend pas de la fusion.
fn action_selon_la_spec(l: usize, t: usize, largeur: usize, u: &[kvu::heritage::su2_hb::Q]) -> f64 {
    use kvu::heritage::su2_hb::{qdag, qmul};
    let (l3, v1) = (l * l * l, l * l * l * t);
    let v = 2 * v1;
    let lien = |mu: usize, r: usize, tt: usize, x: [usize; 3]| {
        u[mu * v + r * v1 + tt * l3 + (x[0] * l + x[1]) * l + x[2]]
    };
    let haut = |x: [usize; 3], d: usize| {
        let mut y = x;
        y[d - 1] = (y[d - 1] + 1) % l;
        y
    };
    // lien spatial (x, d) dans A : ses deux extrémités dans la tranche x1 < largeur ; face supérieure de r : A pris à r + 1
    let dans_a = |x: [usize; 3], d: usize| x[0] < largeur && haut(x, d)[0] < largeur;
    let pi = |r: usize, x: [usize; 3], d: usize| if dans_a(x, d) { 1 - r } else { r };
    // 1 − ½Tr(a b c† d†), produit groupé autrement que dans le module
    let un_moins = |a, b, c, d| 1.0 - qmul(qmul(a, b), qmul(qdag(c), qdag(d)))[0];
    let mut s = 0.0;
    for r in 0..2 {
        for tt in 0..t {
            for x1 in 0..l {
                for x2 in 0..l {
                    for x3 in 0..l {
                        let x = [x1, x2, x3];
                        for nu in 1..=3 {
                            // plaquette (0, ν) ; à t = T − 1, le lien spatial du haut est celui de la face supérieure
                            let (rh, th) = if tt + 1 < t {
                                (r, tt + 1)
                            } else {
                                (pi(r, x, nu), 0)
                            };
                            s += un_moins(
                                lien(0, r, tt, x),
                                lien(nu, rh, th, x),
                                lien(0, r, tt, haut(x, nu)),
                                lien(nu, r, tt, x),
                            );
                        }
                        for mu in 1..=3 {
                            for nu in mu + 1..=3 {
                                let (xm, xn) = (haut(x, mu), haut(x, nu));
                                let bas = un_moins(
                                    lien(mu, r, tt, x),
                                    lien(nu, r, tt, xm),
                                    lien(mu, r, tt, xn),
                                    lien(nu, r, tt, x),
                                );
                                if tt != 0 {
                                    s += bas;
                                } else {
                                    let p = |y, d| pi(r, y, d);
                                    let sup = un_moins(
                                        lien(mu, p(x, mu), 0, x),
                                        lien(nu, p(xm, nu), 0, xm),
                                        lien(mu, p(xn, mu), 0, xn),
                                        lien(nu, p(x, nu), 0, x),
                                    );
                                    s += 0.5 * bas + 0.5 * sup;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    s
}

#[test]
fn table_des_largeurs_intermediaires_contre_action_selon_la_spec() {
    // relecture, objection 6 : contrôle NON circulaire du câblage aux largeurs intermédiaires (les portes e/e2 lisent la table
    // du module). Champ de Haar (aucune symétrie), β = 1 : action du module (table) = action réécrite depuis la spécification.
    for (l, t) in [(4usize, 4usize), (6, 2)] {
        let g = GeoRep::nouveau(l, t, 2).unwrap();
        for larg in 0..=l {
            let mut r = Reseau::pur(
                g,
                Region::tranche(&g, larg),
                &[11 + larg as u64, 29],
                Mutant::Aucun,
            )
            .unwrap();
            r.beta = 1.0;
            r.init(true);
            let (module, spec) = (r.action_geom(0), action_selon_la_spec(l, t, larg, &r.u));
            assert!(
                ((module - spec) / spec).abs() < 1e-12,
                "L = {l}, T = {t}, largeur {larg} : module {module} contre spécification {spec}"
            );
            // la comparaison voit la largeur : la spécification à largeur ± 1 diffère sur le même champ
            for autre in [larg.wrapping_sub(1), larg + 1]
                .into_iter()
                .filter(|&a| a <= l && a != larg)
            {
                let s2 = action_selon_la_spec(l, t, autre, &r.u);
                assert!(
                    (s2 - module).abs() > 1e-3,
                    "L = {l} : largeurs {larg} et {autre} indiscernables"
                );
            }
        }
        // mutant (agrafe de coupure dans la même réplique, invisible à la porte c pour la largeur 2) : rouge
        let mut m =
            Reseau::pur(g, Region::tranche(&g, 2), &[13, 29], Mutant::AgrafeCoupure).unwrap();
        m.beta = 1.0;
        m.init(true);
        assert!(
            (m.action_geom(0) - action_selon_la_spec(l, t, 2, &m.u)).abs() > 1e-3,
            "mutant AgrafeCoupure invisible à L = {l}"
        );
    }
}

#[test]
fn moyenne_et_erreur_sur_les_memes_observations() {
    // relecture c19, M4 : la moyenne portait sur toutes les observations, l'erreur sur les seuls blocs complets
    let (m, e) = kvu::intrication::moyenne_blocs(&[0.0, 0.0, 0.0, 0.0, 100.0], 2);
    assert_eq!(
        (m, e),
        (0.0, 0.0),
        "moyenne et erreur des 4 observations des 2 blocs complets"
    );
    // n multiple du nombre de blocs (seul cas admis par la ligne de commande) : moyenne = Σx/n au bit près
    let x = [0.3, 1.7, -0.2, 0.9, 2.5, 1.1];
    let (m, e) = kvu::intrication::moyenne_blocs(&x, 3);
    assert_eq!(m.to_bits(), (x.iter().sum::<f64>() / 6.0).to_bits());
    assert!(e.is_finite() && e > 0.0);
}

#[test]
fn resultat_non_fini_refuse_avant_complete() {
    use kvu::intrication::cli::{self, Run};
    // relecture c19, M4 : aucun NaN (sérialisé en null) ne peut sortir avec le code 0 et COMPLETE
    let rep = std::env::temp_dir().join(format!("kvu_intrication_nan_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&rep);
    let run = Run::ouvrir(
        &rep,
        "essai",
        &[("geometrie".into(), cli::spec(2, 2, 0))],
        serde_json::json!({}),
    )
    .unwrap();
    let e = cli::conduire(run, |r| {
        r.ligne(&serde_json::json!({"x": f64::NAN, "y": 1.0}))
            .map(|_| 0)
    })
    .unwrap_err();
    assert_eq!(e.code_sortie(), 8, "{e}");
    let m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(rep.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(m["statut"], "echec", "{m}");
    assert!(!rep.join("COMPLETE").exists());
    let _ = std::fs::remove_dir_all(&rep);
}

#[test]
fn doublement_et_balayage_refusent_beta_hors_domaine() {
    use kvu::intrication::portes::ParamsChaine;
    // relecture c19, M7 : β = 1e308 passait le contrôle ; la grille β·i/(n−1) débordait en infini
    let pc = ParamsChaine {
        therm: 0,
        mes: 2,
        sep: 1,
        n_or: 0,
        blocs: 2,
    };
    assert!(portes::doublement(2, 2, 1e308, 3, 3, pc, 1).is_err());
    assert!(portes::balayage(2, 2, 0, 1, 1e7, 3, pc, 1).is_err());
    // relecture c19b, bloquante 10 : domaine admis [10⁻³, 10³], bornes comprises
    for b in [1e-3, 1e3] {
        assert!(portes::balayage(2, 2, 0, 1, b, 3, pc, 1).is_ok(), "β = {b}");
    }
    for b in [0.0009999999999999998, 1000.0000000000001, 1e-20] {
        assert!(
            portes::balayage(2, 2, 0, 1, b, 3, pc, 1).is_err(),
            "β = {b}"
        );
    }
}

#[test]
fn filet_plafond_et_erreur_nulle_independant_du_domaine() {
    use kvu::intrication::portes::{sante_points, PointAlpha};
    use kvu::intrication::{chaine, cli, moyenne_blocs, Flux, Route};
    // relecture c19b, bloquante 10 (b) : HORS du domaine (β = 1e-20, chaîne construite directement, sans `balayage`),
    // e^(−2β|A|) vaut 1, Creutz rejette tout, chaque lien atteint le plafond d'essais et la série est constante. Le filet
    // refuse ce point ; il refuse aussi une erreur nulle seule et accepte un point sain.
    let flux = Flux::paire(1, Route::Alpha, 0, 0).unwrap();
    let mut r = Reseau::depuis_spec_flux(
        &cli::spec(2, 2, 0),
        &cli::spec(2, 2, 1),
        &flux,
        Mutant::Aucun,
    )
    .unwrap();
    let c = chaine(&mut r, 1e-20, 0.5, 0, 2, 1, 0, true).unwrap();
    assert!(c.n_cap > 0, "plafond attendu à β = 1e-20");
    let (m, e) = moyenne_blocs(&c.delta, 2);
    assert_eq!(e, 0.0, "série constante attendue");
    let pt = PointAlpha {
        alpha: 0.5,
        moyenne: m,
        erreur: e,
        mesures: 2,
        plaquette: 0.0,
        plafonds: c.n_cap,
    };
    assert!(sante_points(std::slice::from_ref(&pt))
        .unwrap_err()
        .contains("plafond"));
    let nul = PointAlpha {
        plafonds: 0,
        ..pt.clone()
    };
    assert!(sante_points(&[nul]).unwrap_err().contains("erreur"));
    let sain = PointAlpha {
        plafonds: 0,
        erreur: 0.1,
        ..pt
    };
    assert!(sante_points(&[sain]).is_ok());
}

/// 1000 graines tirées par splitmix64 (générateur fixe, indépendant du module), plus 0, 1, 1071 et u64::MAX.
fn graines_du_test() -> Vec<u64> {
    let mut g = vec![0u64, 1, 1071, u64::MAX];
    let mut s = 0x1234_5678_9ABC_DEF0u64;
    for _ in 0..1000 {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        g.push(z ^ (z >> 31));
    }
    g
}

/// Couples (clé, compteur de départ) des flux : départ chaud (k0, k1, w1_init, w3_init) et bain thermique (k0, k1, 0, w3_hb).
fn couples(flux: &[[kvu::intrication::Flux; 2]]) -> Vec<(u32, u32, u32, u32)> {
    flux.iter()
        .flatten()
        .flat_map(|f| [(f.k0, f.k1, f.w1_init, f.w3_init), (f.k0, f.k1, 0, f.w3_hb)])
        .collect()
}

fn tous_distincts(v: &[(u32, u32, u32, u32)]) -> bool {
    let mut w = v.to_vec();
    w.sort_unstable();
    w.dedup();
    w.len() == v.len()
}

#[test]
fn flux_distincts_par_construction() {
    use kvu::intrication::portes::{plan_balayage, plan_courbe, plan_debit, plan_doublement};
    // relecture c19, bloquante 1 : `doublement --graine 1071` donnait la même clé aux deux répliques (XOR annulé). Pour chaque
    // commande et chaque graine, tous les couples (clé, compteur de départ) d'un run sont distincts ; l'union sur les graines
    // aussi (la graine est un champ du flux, pas un terme mélangé).
    let largeurs: Vec<usize> = (0..8).collect();
    for (commande, plan) in [
        (
            "balayage",
            &(|g| plan_balayage(g, 101, 3))
                as &dyn Fn(u64) -> Result<Vec<[kvu::intrication::Flux; 2]>, String>,
        ),
        ("courbe", &|g| plan_courbe(g, 9, &largeurs)),
        ("doublement", &|g| plan_doublement(g, 9, 9)),
    ] {
        let mut union = Vec::new();
        for g in graines_du_test() {
            let c = couples(&plan(g).unwrap());
            assert!(tous_distincts(&c), "{commande}, graine {g}");
            union.extend(c);
        }
        assert!(
            tous_distincts(&union),
            "{commande} : collision entre graines"
        );
    }
    // bornes : nombre de points maximal, largeur maximale de L = 512, graines extrêmes
    for g in [0, 1071, u64::MAX] {
        assert!(
            tous_distincts(&couples(&plan_doublement(g, 32767, 32767).unwrap())),
            "doublement maximal, graine {g}"
        );
        let c: Vec<usize> = (0..512).collect();
        assert!(
            tous_distincts(&couples(&plan_courbe(g, 3, &c).unwrap())),
            "courbe maximale, graine {g}"
        );
    }
    assert!(
        plan_balayage(0, 32769, 0).is_err() && plan_balayage(0, 3, 4096).is_err(),
        "point 32768 ou largeur 4096 hors champ"
    );
    // debit (graine fixe) : ses deux répliques ; un réseau refuse deux répliques de même flux
    for l in (2..=512).step_by(2) {
        assert!(tous_distincts(&couples(&[plan_debit(l / 2).unwrap()])));
    }
    let g = GeoRep::nouveau(2, 2, 2).unwrap();
    assert!(Reseau::nouveau(
        g,
        Region::tranche(&g, 0),
        Region::tranche(&g, 1),
        &[5, 5],
        Mutant::Aucun
    )
    .is_err());
}

#[test]
fn besoin_memoire_pur_et_refus_deterministe() {
    use kvu::intrication::cli;
    use kvu::intrication::portes::ParamsChaine;
    use kvu::machine::admission;
    // relecture c19, mineure 9 et majeure 6 : besoin calculé sans rien allouer, décidé contre un plafond DONNÉ (pas la RAM de
    // la machine) ; il croît avec les mesures ET les blocs ET les points
    let p = ParamsChaine {
        therm: 100,
        mes: 400,
        sep: 1,
        n_or: 3,
        blocs: 20,
    };
    let b60 = cli::besoin_memoire(60, 60, 8, &p, 9).unwrap();
    assert!(b60 > 250 << 30, "L = T = 60, 8 chaînes : {b60} octets");
    let e = admission::decider(
        "intrication balayage L = 60",
        b60 as f64 / (1u64 << 30) as f64,
        97.0,
        25.0,
    )
    .unwrap_err();
    assert_eq!(e.code_sortie(), 5);
    let petit = cli::besoin_memoire(2, 2, 1, &p, 3).unwrap();
    assert!(admission::decider("petit", petit as f64 / (1u64 << 30) as f64, 97.0, 25.0).is_ok());
    let gros_blocs = ParamsChaine {
        mes: 67_108_864,
        blocs: 67_108_864,
        ..p
    };
    let sans_blocs = ParamsChaine {
        mes: 67_108_864,
        blocs: 2,
        ..p
    };
    let d = cli::besoin_memoire(2, 2, 1, &gros_blocs, 3).unwrap()
        - cli::besoin_memoire(2, 2, 1, &sans_blocs, 3).unwrap();
    assert!(
        d >= (67_108_864 - 2) * 8,
        "tableau des moyennes de blocs non compté : {d}"
    );
    assert!(
        cli::besoin_memoire(2, 2, 1, &p, 32767).unwrap()
            > cli::besoin_memoire(2, 2, 1, &p, 3).unwrap() + 32000 * 512
    );
}

#[test]
fn erreur_de_synchronisation_close_en_echec() {
    use kvu::intrication::cli::{self, Run};
    // relecture c19, majeure 8 : `sync_all().ok()` ignorait une erreur d'E/S (EIO) puis rendait 0 et COMPLETE
    let rep = std::env::temp_dir().join(format!("kvu_intrication_sync_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&rep);
    let mut run = Run::ouvrir(
        &rep,
        "essai",
        &[("geometrie".into(), cli::spec(2, 2, 0))],
        serde_json::json!({}),
    )
    .unwrap();
    run.ligne(&serde_json::json!({"x": 1.0})).unwrap();
    let e = run
        .terminer_avec(0, |_| Err(std::io::Error::other("EIO simulé")))
        .unwrap_err();
    assert!(e.to_string().contains("synchronisation"), "{e}");
    let m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(rep.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(m["statut"], "echec", "{m}");
    assert!(!rep.join("COMPLETE").exists());
    let _ = std::fs::remove_dir_all(&rep);
}

#[test]
fn besoin_memoire_compte_series_et_construction_ensemble() {
    use kvu::intrication::cli;
    use kvu::intrication::portes::ParamsChaine;
    // relecture c19b, majeure 11 : la route β réserve ses deux séries AVANT de construire son réseau ; l'estimation doit
    // compter les séries EN PLUS de la construction (le max des deux ne majorait pas : L = T = 32, mes = 2²⁶)
    let p = |mes| ParamsChaine {
        therm: 0,
        mes,
        sep: 1,
        n_or: 0,
        blocs: 2,
    };
    let gros = cli::besoin_memoire(32, 32, 1, &p(1 << 26), 3).unwrap();
    let petit = cli::besoin_memoire(32, 32, 1, &p(2), 3).unwrap();
    assert!(
        gros - petit >= 16 * ((1 << 26) - 2),
        "séries non ajoutées à la construction : {gros} − {petit}"
    );
}
