//! Domaine admis des options de `kvu intrication` (relecture c19b, méthode commune avec le module tube) : pour chaque option,
//! les bornes (comprises) et des valeurs intérieures sont ADMISES, et juste dehors de chaque borne, zéro, NaN, infinis,
//! maximum du type et au-delà du type sont REFUSÉS en code 2, par `cli::valider` : le contrôle UNIQUE que `executer` appelle
//! avant l'admission mémoire et toute écriture. Rien n'est calculé ici (une valeur admise n'est pas exécutée).
//! Le tableau `cli::DOMAINE` est repris mot pour mot par l'aide et par la note.

use kvu::intrication::cli;
use std::path::Path;

const U64MAX: &str = "18446744073709551615";

/// Verdict de la ligne de commande `args` : Ok si admise ; Err(code) sinon (clap refuse au-delà du type : code d'usage 2).
fn verdict(args: &[String]) -> Result<(), i32> {
    match cli::commande().try_get_matches_from(args) {
        Err(_) => Err(2),
        Ok(m) => cli::valider(&m).map_err(|e| e.code_sortie()),
    }
}

/// Ligne de base valide d'une sous-commande, dont on remplace (ou ajoute) des options.
fn ligne(sous: &str, remplace: &[(&str, &str)]) -> Vec<String> {
    let base: &[(&str, &str)] = match sous {
        "balayage" => &[
            ("--l", "2"),
            ("--t", "2"),
            ("--l0", "0"),
            ("--l1", "1"),
            ("--beta", "1.0"),
            ("--nalpha", "3"),
            ("--therm", "5"),
            ("--mes", "20"),
            ("--sep", "1"),
            ("--blocs", "4"),
            ("--or", "1"),
            ("--graine", "1"),
            ("--sortie", "r"),
        ],
        "courbe" => &[
            ("--l", "4"),
            ("--t", "2"),
            ("--largeurs", "1"),
            ("--beta", "1.0"),
            ("--mes", "20"),
            ("--blocs", "4"),
            ("--sortie", "r"),
        ],
        "doublement" => &[
            ("--l", "2"),
            ("--t", "2"),
            ("--beta", "1.0"),
            ("--mes", "20"),
            ("--blocs", "4"),
            ("--sortie", "r"),
        ],
        "debit" => &[("--l", "2"), ("--t", "2"), ("--pas", "1"), ("--or", "0")],
        _ => unreachable!(),
    };
    let mut o: Vec<(String, String)> = base
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    for (k, v) in remplace {
        match o.iter_mut().find(|(x, _)| x == k) {
            Some(e) => e.1 = v.to_string(),
            None => o.push((k.to_string(), v.to_string())),
        }
    }
    let mut a = vec!["intrication".to_string(), sous.to_string()];
    a.extend(o.into_iter().map(|(k, v)| format!("{k}={v}")));
    a
}

/// (sous-commande, options remplacées, admise ?)
type Cas = (&'static str, Vec<(&'static str, &'static str)>, bool);

fn cas() -> Vec<Cas> {
    let mut v: Vec<Cas> = Vec::new();
    let mut une = |sous: &'static str,
                   opt: &'static str,
                   dedans: &[&'static str],
                   dehors: &[&'static str]| {
        for x in dedans {
            v.push((sous, vec![(opt, *x)], true));
        }
        for x in dehors {
            v.push((sous, vec![(opt, *x)], false));
        }
    };
    // --l, --t (pairs ≥ 2) ; la capacité est éprouvée plus bas (paire L = T)
    une(
        "balayage",
        "--l",
        &["2", "4", "80"],
        &["0", "1", "3", U64MAX],
    );
    une(
        "balayage",
        "--t",
        &["2", "4", "80"],
        &["0", "1", "3", U64MAX],
    );
    // --l0, --l1 : 0 ≤ l ≤ L, différents
    une("balayage", "--l1", &["2"], &["0", "3", U64MAX]);
    une("balayage", "--l0", &["2"], &["1", "3", U64MAX]);
    // --beta : fini, [0,001 ; 1000]
    une(
        "balayage",
        "--beta",
        &[
            "0.001",
            "1000",
            "2.3",
            "0.0010000000000000002",
            "999.9999999999999",
        ],
        &[
            "0.0009999999999999998",
            "1000.0000000000001",
            "0",
            "-0",
            "-1",
            "NaN",
            "inf",
            "-inf",
            "1e-20",
            "5e-324",
            "1.7976931348623157e308",
        ],
    );
    // --nalpha, --nbeta : impairs, [3 ; 32767]
    une(
        "balayage",
        "--nalpha",
        &["3", "9", "32767"],
        &["0", "1", "2", "4", "32768", "32769", U64MAX],
    );
    une(
        "doublement",
        "--nbeta",
        &["3", "32767"],
        &["0", "1", "2", "32769", U64MAX],
    );
    une("doublement", "--nalpha", &["3", "32767"], &["2", "32769"]);
    // budget : therm + mes × sep < 2³² (therm 5, mes 20, sep 1 dans la ligne de base)
    une(
        "balayage",
        "--therm",
        &["0", "4294967275"],
        &["4294967276", U64MAX],
    );
    une(
        "balayage",
        "--sep",
        &["1", "214748364"],
        &["0", "214748365", U64MAX],
    );
    // --mes : multiple de --blocs (4), ≥ --blocs, budget
    une(
        "balayage",
        "--mes",
        &["4", "20", "1073741820"],
        &["0", "1", "2", "6", "4294967292", U64MAX],
    );
    // --blocs : 2 ≤ blocs ≤ mes (20), divise mes
    une(
        "balayage",
        "--blocs",
        &["2", "4", "5", "10", "20"],
        &["0", "1", "3", "40", U64MAX],
    );
    // --or : [0 ; 1000]
    une("balayage", "--or", &["0", "1000"], &["1001", U64MAX]);
    une("debit", "--or", &["0", "1000"], &["1001", U64MAX]);
    // --graine : tout u64
    une(
        "balayage",
        "--graine",
        &["0", U64MAX],
        &["18446744073709551616", "-1"],
    );
    // --largeurs (L = 4) : 0 ≤ l < L, distinctes
    une(
        "courbe",
        "--largeurs",
        &["0", "3", "0,1,2,3"],
        &["4", "1,1", "0,4", U64MAX],
    );
    // --pas : ≥ 1, 1 + pas < 2³²
    une(
        "debit",
        "--pas",
        &["1", "4294967294"],
        &["0", "4294967295", U64MAX],
    );
    // capacité des plaquettes (2·(6V + 6L³) ≤ 2³⁰) : L = T = 80 dedans, 82 dehors
    v.push(("debit", vec![("--l", "80"), ("--t", "80")], true));
    v.push(("debit", vec![("--l", "82"), ("--t", "82")], false));
    v.push(("balayage", vec![("--l", "82"), ("--t", "82")], false));
    v
}

#[test]
fn domaine_parametre_bornes_et_juste_dehors() {
    let mut echecs = Vec::new();
    for (sous, remplace, admise) in cas() {
        let a = ligne(sous, &remplace);
        let v = verdict(&a);
        let ok = if admise { v.is_ok() } else { v == Err(2) };
        if !ok {
            echecs.push(format!(
                "{sous} {remplace:?} : attendu {}, obtenu {v:?}",
                if admise { "admise" } else { "code 2" }
            ));
        }
    }
    assert!(
        echecs.is_empty(),
        "{} cas faux :\n{}",
        echecs.len(),
        echecs.join("\n")
    );
}

#[test]
fn le_tableau_du_domaine_est_dans_l_aide_et_dans_la_note() {
    let note = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/v0.2/notes/intrication.md"),
    )
    .unwrap();
    let aide = cli::commande()
        .find_subcommand_mut("balayage")
        .unwrap()
        .render_long_help()
        .to_string();
    for (o, i, _) in cli::DOMAINE {
        assert!(note.contains(i), "note : intervalle de {o} absent : {i}");
        assert!(aide.contains(i), "aide : intervalle de {o} absent : {i}");
    }
}
