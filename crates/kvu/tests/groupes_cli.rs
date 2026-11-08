//! Tests du registre des groupes par la ligne de commande (WP5) : refus de Sp(4)/G₂ sans certificat (G-CERT), format des lignes,
//! sous-ensembles de la batterie, refus du certificat d'une batterie partielle. Aucun GPU ; les verdicts physiques complets
//! (G-SC, G-WC, mutants β × 1,05) sont ceux de `kvu groupe gates --groupe su2`, pas de ces tests de plomberie.

use std::process::{Command, Output};

fn kvu(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_kvu")).args(args).output().expect("lancement de kvu")
}

fn out(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

#[test]
fn liste_donne_su2_seul() {
    let o = kvu(&["groupe", "liste"]);
    assert_eq!(o.status.code(), Some(0));
    // WP16 : sp4 a maintenant une fiche au registre (assertion v3.1 « su2 » modifiée, signalée au journal)
    assert_eq!(out(&o).trim(), "su2\nsp4");
}

#[test]
fn sp4_et_g2_refuses_sans_certificat() {
    // WP16 : sp4 a une fiche (donc n'est plus refusé par `gates`) ; seul g2 reste refusé. Assertion modifiée, signalée au journal.
    #[allow(clippy::single_element_loop)]
    for g in ["g2"] {
        let o = kvu(&["groupe", "gates", "--groupe", g]);
        assert_eq!(o.status.code(), Some(2), "{g} : {}", err(&o));
        assert!(err(&o).contains("refusé"), "{}", err(&o));
        assert!(out(&o).is_empty(), "aucune ligne de gate ne doit être émise pour {g}");
        let c = kvu(&["groupe", "certificat", "--groupe", g, "--dossier", env!("CARGO_TARGET_TMPDIR")]);
        assert_eq!(c.status.code(), Some(2), "{}", err(&c));
    }
}

#[test]
fn algebre_et_haar_vertes_avec_lignes_bien_formees_et_json() {
    let o = kvu(&["groupe", "gates", "--groupe", "su2", "--seul", "G-ALG,G-HAAR", "--tirages", "200000", "--sans-mutants", "--json"]);
    let s = out(&o);
    assert_eq!(o.status.code(), Some(0), "{s}\n{}", err(&o));
    let mut attendus = vec!["G-ALG-1", "G-ALG-2", "G-ALG-3", "G-HAAR-1", "G-HAAR-2", "G-HAAR-3", "G-HAAR-3-CHAMP"];
    for l in s.lines() {
        if l.starts_with('{') {
            let j: serde_json::Value = serde_json::from_str(l).expect("JSON final");
            assert_eq!(j["tout_vert"], true);
            assert_eq!(j["fail"], 0);
            assert_eq!(j["skip"], 0);
            continue;
        }
        let mut it = l.splitn(3, ' ');
        let (v, id) = (it.next().unwrap(), it.next().unwrap());
        assert!(["PASS", "FAIL", "SKIP", "INFO"].contains(&v), "ligne mal formée : {l}");
        if v == "PASS" {
            attendus.retain(|a| *a != id);
        }
    }
    assert!(attendus.is_empty(), "gates non PASS : {attendus:?}\n{s}");
}

#[test]
fn certificat_refuse_pour_une_batterie_partielle() {
    let rep = format!("{}/groupes_cli_cert_{}", env!("CARGO_TARGET_TMPDIR"), std::process::id());
    for extra in [vec!["--seul", "G-ALG"], vec!["--sans-mutants"], vec!["--tirages", "1000"]] {
        let mut a = vec!["groupe", "gates", "--groupe", "su2", "--certifier", rep.as_str()];
        a.extend(extra.iter());
        let o = kvu(&a);
        assert_eq!(o.status.code(), Some(2), "{}", err(&o));
    }
    assert!(!std::path::Path::new(&rep).exists(), "aucun certificat ne doit être écrit");
    let c = kvu(&["groupe", "certificat", "--groupe", "su2", "--dossier", rep.as_str()]);
    assert_eq!(c.status.code(), Some(2), "{}", err(&c));
}

#[test]
fn derive_haar_numerique_pente_un_quart() {
    let o = kvu(&["groupe", "derive", "--groupe", "su2", "--tirages", "400000"]);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    let s = out(&o);
    assert!(s.contains("⟨x²⟩ = 0.2"), "{s}");
    assert!(s.contains("a_attendu (u + 4u⁵, v3.1) recalculé = 0.250202005964"), "{s}");
}
