//! Tests de plomberie de Sp(4) au registre (WP16) par la ligne de commande : liste, plan, gates partielles, refus du certificat d'une batterie
//! partielle, refus de `gen` sans certificat, lecture des séries (SKIP sans valeur physique). Les verdicts physiques complets sont ceux de
//! `kvu groupe gates --groupe sp4` (certificat `validation/groupes/sp4.json`), pas de ces tests. Aucun GPU.

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
fn tmp(nom: &str) -> String {
    format!("{}/groupes_sp4_{}_{}", env!("CARGO_TARGET_TMPDIR"), nom, std::process::id())
}

#[test]
fn sp4_au_registre() {
    let o = kvu(&["groupe", "liste"]);
    assert_eq!(o.status.code(), Some(0));
    assert_eq!(out(&o).trim(), "su2\nsp4");
}

#[test]
fn plan_liste_toutes_les_chaines() {
    let d = tmp("plan");
    let o = kvu(&["groupe", "plan", "--groupe", "sp4", "--dossier", &d, "--binaire", "/x/kvu"]);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    let s = out(&o);
    let lignes: Vec<&str> = s.lines().collect();
    // UNIT + 16 REG + 16 REGM + 12 WC (3 × 4) + 6 SC (3 × 2)
    assert_eq!(lignes.len(), 1 + 16 + 16 + 12 + 6, "{s}");
    assert!(lignes.iter().all(|l| l.starts_with("/x/kvu groupe chaine --groupe sp4 --id ") && l.ends_with(&format!("--dossier {d}"))));
    for id in ["UNIT", "REG-01", "REG-16", "REGM-08", "WC-b50", "WCM105-b400", "WCM102-b100", "SC-b0.1", "SCM105-b0.2", "SCM102-b0.1"] {
        assert!(lignes.iter().any(|l| l.contains(&format!("--id {id} "))), "chaîne {id} absente");
    }
    // les plus coûteuses d'abord : REG (L = 16) avant WC (L = 8) avant SC (L = 6)
    let pos = |id: &str| lignes.iter().position(|l| l.contains(&format!("--id {id} "))).unwrap();
    assert!(pos("REG-01") < pos("WC-b50") && pos("WC-b50") < pos("SC-b0.1"));
    let o = kvu(&["groupe", "plan", "--groupe", "sp4", "--dossier", &d, "--sans-mutants", "--sans-info"]);
    assert_eq!(out(&o).lines().count(), 1 + 16 + 4 + 2);
    let o = kvu(&["groupe", "plan", "--groupe", "g2", "--dossier", &d]);
    assert_eq!(o.status.code(), Some(2));
}

#[test]
fn algebre_et_haar_sp4_vertes() {
    let o = kvu(&["groupe", "gates", "--groupe", "sp4", "--seul", "Fiche,G-ALG,G-HAAR,G-SC-SERIE", "--tirages", "200000", "--sans-mutants", "--sans-calcul", "--json"]);
    let s = out(&o);
    assert_eq!(o.status.code(), Some(0), "{s}\n{}", err(&o));
    let mut attendus = vec!["Fiche", "G-ALG-1", "G-ALG-2", "G-ALG-3", "G-ALG-DIM", "G-HAAR-1", "G-HAAR-2", "G-HAAR-3", "G-HAAR-3-CHAMP", "G-SC-SERIE"];
    for l in s.lines() {
        if l.starts_with('{') {
            let j: serde_json::Value = serde_json::from_str(l).expect("JSON final");
            assert_eq!(j["tout_vert"], true);
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
fn mutants_algebre_et_haar_sont_rouges() {
    let o = kvu(&["groupe", "gates", "--groupe", "sp4", "--seul", "G-MUT-ALG,G-MUT-HAAR", "--tirages", "200000", "--sans-calcul"]);
    let s = out(&o);
    assert_eq!(o.status.code(), Some(0), "{s}\n{}", err(&o));
    for id in ["G-MUT-ALG", "G-MUT-HAAR2", "G-MUT-HAAR3"] {
        assert!(s.lines().any(|l| l.starts_with(&format!("PASS {id} mutant ROUGE"))), "{id} : {s}");
    }
}

#[test]
fn sans_serie_les_gates_a_chaines_sont_skip_sans_valeur_physique() {
    let d = tmp("vide");
    let o = kvu(&["groupe", "gates", "--groupe", "sp4", "--seul", "G-SC,G-WC,G-REG,G-UNIT,G-MUT-SC", "--sans-calcul", "--chaines", &d]);
    let s = out(&o);
    assert_eq!(o.status.code(), Some(8), "{s}\n{}", err(&o));
    for id in ["G-SC", "G-WC", "G-REG", "G-UNIT", "G-MUT-SC"] {
        let l = s.lines().find(|l| l.split(' ').nth(1) == Some(id)).unwrap_or_else(|| panic!("{id} absent : {s}"));
        assert!(l.starts_with("SKIP "), "{l}");
        assert!(!l.contains("z ="), "une ligne SKIP ne doit afficher aucune valeur physique : {l}");
    }
    assert!(!std::path::Path::new(&d).join("SC-b0.1.serie").exists(), "--sans-calcul ne simule rien");
}

#[test]
fn certificat_refuse_pour_une_batterie_partielle() {
    let rep = tmp("cert");
    for extra in [vec!["--seul", "G-ALG"], vec!["--sans-mutants"], vec!["--tirages", "1000"]] {
        let mut a = vec!["groupe", "gates", "--groupe", "sp4", "--certifier", rep.as_str()];
        a.extend(extra.iter());
        let o = kvu(&a);
        assert_eq!(o.status.code(), Some(2), "{}", err(&o));
    }
    assert!(!std::path::Path::new(&rep).exists(), "aucun certificat ne doit être écrit");
}

#[test]
fn gen_refuse_sans_certificat_et_gpu_refuse() {
    let rep = tmp("gen");
    let o = kvu(&["gen", "--groupe", "sp4", "--L", "4", "--nmeas", "2", "--outdir", &format!("{rep}/x"), "--certificats", &rep]);
    assert_eq!(o.status.code(), Some(2), "{}", err(&o));
    assert!(err(&o).contains("refusé"), "{}", err(&o));
    assert!(!std::path::Path::new(&format!("{rep}/x")).exists());
    let g = kvu(&["gen", "--groupe", "g2", "--outdir", &format!("{rep}/y"), "--certificats", &rep]);
    assert_eq!(g.status.code(), Some(2));
    let o = kvu(&["groupe", "gates", "--groupe", "sp4", "--backend", "hip", "--seul", "Fiche"]);
    assert_eq!(o.status.code(), Some(2), "{}", err(&o));
}

#[test]
fn gen_ecrit_un_ensemble_avec_le_certificat_du_depot() {
    let certs = format!("{}/validation/groupes", env!("CARGO_MANIFEST_DIR"));
    let c = kvu(&["groupe", "certificat", "--groupe", "sp4", "--dossier", &certs]);
    assert_eq!(c.status.code(), Some(0), "{}{}", out(&c), err(&c));
    let d = tmp("gen_ok");
    let o = kvu(&["gen", "--groupe", "sp4", "--L", "4", "--beta", "12", "--graine", "7", "--therm", "100", "--nmeas", "6", "--skip", "2", "--dump-every", "3", "--outdir", &d, "--certificats", &certs]);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    for f in ["COMPLETE", "manifest.json", "measurements.csv", "cfg_000003.npy", "cfg_000006.npy"] {
        assert!(std::path::Path::new(&d).join(f).exists(), "{f} absent");
    }
    let csv = std::fs::read_to_string(format!("{d}/measurements.csv")).unwrap();
    assert_eq!(csv.lines().count(), 1 + 7, "en-tête + indices 0…6");
    let p: f64 = csv.lines().nth(2).unwrap().split(',').nth(1).unwrap().parse().unwrap();
    assert!(p > 0.7 && p < 0.85, "plaquette à β = 12 : {p}");
    let m: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{d}/manifest.json")).unwrap()).unwrap();
    assert_eq!(m["schema"], "kvu-gen/1");
    assert_eq!(m["pas_total"], 100 + 12);
    // NPY complex128 (4, 4, 4, 4, 4, 4, 4) : 4 L⁴ liens de 16 complexes
    let npy = std::fs::read(format!("{d}/cfg_000006.npy")).unwrap();
    let hl = u16::from_le_bytes([npy[8], npy[9]]) as usize;
    assert_eq!(npy.len(), 10 + hl + 4 * 256 * 16 * 16);
    // un répertoire de sortie existant est refusé
    let o2 = kvu(&["gen", "--groupe", "sp4", "--L", "4", "--nmeas", "1", "--outdir", &d, "--certificats", &certs]);
    assert_eq!(o2.status.code(), Some(2), "{}", err(&o2));
}
