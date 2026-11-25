//! `kvu intrication …` par le binaire `kvu` (révisions REV2 du 11/11/2026) : refus d'usage AVANT tout calcul (code 2, ou code 5
//! pour la mémoire ; aucun répertoire de run créé), bornes de chaque option numérique, étiquette EXPÉRIMENTALE (aide, manifeste,
//! stderr), avertissement de la tranche entière, exemples de la note. Les commandes qui DOIVENT être refusées ont un délai
//! extérieur (relecture c19b, mineure 13) : au-delà, SEUL le processus fils est tué et le test est rouge ; le succès d'un test
//! ne dépend jamais du délai (un refus est immédiat). Les commandes qui calculent attendent leur fin.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Délai extérieur des commandes qui doivent être refusées (un refus prend quelques millisecondes).
const DELAI_REFUS: Duration = Duration::from_secs(120);

const AVERTISSEMENT: &str = "module expérimental : interface et formats susceptibles de changer ; résultats à confirmer indépendamment";

/// Nom de la surcharge de la garde SSD de production (`src/stockage/garde.rs`).
const VAR_SEUIL_SSD: &str = "KEVINOTRON_SSD_MIN_GO";

/// Seuil imposé aux sous-processus des tests qui calculent : garde inoffensive, donc tests indépendants de la place disque.
const SEUIL_SSD_TEST: &str = "0";

/// Environnement d'un sous-processus : les paires demandées, puis `KEVINOTRON_SSD_MIN_GO=0` EN DERNIER (la dernière écriture
/// gagne : la valeur du test l'emporte sur toute paire antérieure et sur la variable héritée de `cargo test`).
///
/// Exception : si le test impose LUI-MÊME `KEVINOTRON_SSD_MIN_GO` (test `garde_ssd_…`, qui vérifie que la garde de
/// production refuse un disque plein), sa valeur est conservée et le seuil 0 n'est PAS ajouté — sinon le sous-processus
/// recevrait 0 et écrirait, et le test ne prouverait plus rien.
fn env_avec_seuil_ssd(env: &[(&str, &str)]) -> Vec<(String, String)> {
    let mut e: Vec<(String, String)> = env
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    if !e.iter().any(|(k, _)| k == VAR_SEUIL_SSD) {
        e.push((VAR_SEUIL_SSD.to_string(), SEUIL_SSD_TEST.to_string()));
    }
    e
}

/// Répertoire de travail neuf du test (sous TMPDIR : la garde SSD de kvu exige un disque, pas un tmpfs).
fn dossier(nom: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("kvu_intrication_cli_{}_{nom}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct Sortie {
    rc: Option<i32>,
    stdout: String,
    stderr: String,
}

/// Lance `kvu args…` (variables d'environnement `env` en plus) et attend sa fin.
fn kvu(args: &[&str], env: &[(&str, &str)]) -> Sortie {
    let mut c = Command::new(env!("CARGO_BIN_EXE_kvu"));
    c.args(args).stdin(Stdio::null());
    for (k, v) in env_avec_seuil_ssd(env) {
        c.env(k, v);
    }
    let o = c.output().expect("lancement de kvu");
    Sortie {
        rc: o.status.code(),
        stdout: String::from_utf8_lossy(&o.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&o.stderr).into_owned(),
    }
}

/// Lance `kvu args…` avec un délai : au-delà, le fils (et lui seul) est tué et `rc` vaut None.
fn kvu_borne(args: &[&str], env: &[(&str, &str)], delai: Duration) -> Sortie {
    static NUMERO: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let d = dossier(&format!(
        "borne_{}",
        NUMERO.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    ));
    let (fo, fe) = (d.join("stdout.txt"), d.join("stderr.txt"));
    let mut c = Command::new(env!("CARGO_BIN_EXE_kvu"));
    c.args(args).stdin(Stdio::null());
    c.stdout(std::fs::File::create(&fo).unwrap())
        .stderr(std::fs::File::create(&fe).unwrap());
    for (k, v) in env_avec_seuil_ssd(env) {
        c.env(k, v);
    }
    let mut fils = c.spawn().expect("lancement de kvu");
    let debut = Instant::now();
    let rc = loop {
        if let Some(st) = fils.try_wait().unwrap() {
            break st.code();
        }
        if debut.elapsed() > delai {
            let _ = fils.kill();
            let _ = fils.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let s = Sortie {
        rc,
        stdout: std::fs::read_to_string(&fo).unwrap_or_default(),
        stderr: std::fs::read_to_string(&fe).unwrap_or_default(),
    };
    let _ = std::fs::remove_dir_all(&d);
    s
}

/// Une commande refusée : code attendu, motif attendu sur stderr, aucun répertoire de run créé.
fn refus(nom: &str, args: &[&str], env: &[(&str, &str)], rc: i32, motif: &str) {
    let d = dossier(nom);
    let run = d.join("run");
    let r = run.to_str().unwrap().to_string();
    let mut tout: Vec<&str> = args.to_vec();
    if args.get(1) != Some(&"debit") {
        tout.extend(["--sortie", r.as_str()]);
    }
    let s = kvu_borne(&tout, env, DELAI_REFUS);
    assert_eq!(
        s.rc,
        Some(rc),
        "{nom} : {tout:?}\nstdout : {}\nstderr : {}",
        s.stdout,
        s.stderr
    );
    assert!(
        s.stderr.contains(motif),
        "{nom} : motif « {motif} » absent de stderr : {}",
        s.stderr
    );
    assert!(
        !run.exists(),
        "{nom} : un répertoire de run a été créé malgré le refus"
    );
    let _ = std::fs::remove_dir_all(&d);
}

/// Balayage minimal valide (L = T = 2), dont on remplace une option.
fn balayage(remplace: &[(&str, &str)]) -> Vec<String> {
    let mut opts: Vec<(String, String)> = [
        ("--l", "2"),
        ("--t", "2"),
        ("--l0", "0"),
        ("--l1", "1"),
        ("--beta", "1.0"),
        ("--nalpha", "3"),
        ("--therm", "5"),
        ("--mes", "20"),
        ("--blocs", "4"),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    for (k, v) in remplace {
        match opts.iter_mut().find(|(o, _)| o == k) {
            Some(o) => o.1 = v.to_string(),
            None => opts.push((k.to_string(), v.to_string())),
        }
    }
    let mut a = vec!["intrication".to_string(), "balayage".to_string()];
    for (k, v) in opts {
        a.push(format!("{k}={v}"));
    }
    a
}

fn refus_balayage(nom: &str, remplace: &[(&str, &str)], motif: &str) {
    let a = balayage(remplace);
    let r: Vec<&str> = a.iter().map(String::as_str).collect();
    refus(nom, &r, &[], 2, motif);
}

#[test]
fn refus_points_hors_bornes() {
    // impair, ≥ 3 et ≤ 32767 (champ de 15 bits du compteur ; tableaux proportionnels aux points bornés) — objections 1 et B2
    for n in ["10", "0", "1", "18446744073709551615", "32769"] {
        refus_balayage(
            &format!("nalpha{n}"),
            &[("--nalpha", n)],
            &format!("--nalpha = {n}"),
        );
    }
    refus(
        "nbeta4",
        &[
            "intrication",
            "doublement",
            "--l",
            "2",
            "--t",
            "2",
            "--beta",
            "1.0",
            "--nbeta",
            "4",
            "--mes",
            "20",
            "--blocs",
            "4",
        ],
        &[],
        2,
        "--nbeta = 4",
    );
    refus(
        "nbeta_max",
        &[
            "intrication",
            "doublement",
            "--l",
            "2",
            "--t",
            "2",
            "--beta",
            "1.0",
            "--nbeta",
            "18446744073709551615",
        ],
        &[],
        2,
        "--nbeta = 18446744073709551615",
    );
    refus(
        "courbe_nalpha2",
        &[
            "intrication",
            "courbe",
            "--l",
            "2",
            "--t",
            "2",
            "--beta",
            "1.0",
            "--largeurs",
            "0",
            "--nalpha",
            "2",
        ],
        &[],
        2,
        "--nalpha = 2",
    );
}

#[test]
fn refus_beta_hors_domaine() {
    // fini, > 0 et ≤ 10⁶ : aucune grandeur intermédiaire infinie (grille de doublement comprise) — objections 3 et M7
    for b in ["0", "-1", "NaN", "inf", "1e308", "1e7"] {
        refus_balayage(&format!("beta{b}"), &[("--beta", b)], "β fini, > 0 et ≤");
    }
    refus(
        "doublement_beta",
        &[
            "intrication",
            "doublement",
            "--l",
            "2",
            "--t",
            "2",
            "--beta=1e308",
            "--nalpha",
            "3",
            "--nbeta",
            "3",
            "--mes",
            "2",
            "--blocs",
            "2",
        ],
        &[],
        2,
        "β fini, > 0 et ≤",
    );
}

#[test]
fn refus_statistiques_invalides() {
    // --sep ≥ 1, 2 ≤ --blocs ≤ --mes, --mes multiple de --blocs (moyenne et erreur sur les MÊMES observations) — M4
    refus_balayage("mes0", &[("--mes", "0")], "--mes");
    refus_balayage("mes1", &[("--mes", "1"), ("--blocs", "2")], "--mes");
    refus_balayage("sep0", &[("--sep", "0")], "--sep");
    refus_balayage("blocs1", &[("--blocs", "1")], "--blocs");
    refus_balayage("blocs0", &[("--blocs", "0")], "--blocs");
    refus_balayage(
        "mes5_blocs2",
        &[("--mes", "5"), ("--blocs", "2")],
        "--blocs",
    );
    refus_balayage(
        "blocs_sup_mes",
        &[("--mes", "20"), ("--blocs", "40")],
        "--blocs",
    );
}

#[test]
fn refus_budget_des_compteurs_philox() {
    // therm + mes × sep < 2³² balayages par chaîne (le compteur de balayage ne reboucle jamais) — M5
    refus_balayage(
        "therm",
        &[("--therm", "4294967295"), ("--mes", "2"), ("--blocs", "2")],
        "balayages par chaîne",
    );
    refus_balayage(
        "mes_sep",
        &[
            ("--therm", "0"),
            ("--mes", "2147483648"),
            ("--sep", "2"),
            ("--blocs", "2"),
        ],
        "balayages par chaîne",
    );
    refus_balayage(
        "produit_deborde",
        &[
            ("--mes", "18446744073709551614"),
            ("--sep", "3"),
            ("--blocs", "2"),
        ],
        "balayages par chaîne",
    );
    refus(
        "debit_pas",
        &[
            "intrication",
            "debit",
            "--l",
            "2",
            "--t",
            "2",
            "--pas",
            "4294967295",
            "--or",
            "0",
        ],
        &[],
        2,
        "balayages par chaîne",
    );
    refus(
        "debit_pas0",
        &["intrication", "debit", "--l", "2", "--t", "2", "--pas", "0"],
        &[],
        2,
        "--pas",
    );
}

#[test]
fn refus_or_hors_borne() {
    refus_balayage("or", &[("--or", "1001")], "--or");
    refus(
        "debit_or",
        &[
            "intrication",
            "debit",
            "--l",
            "2",
            "--t",
            "2",
            "--or",
            "18446744073709551615",
        ],
        &[],
        2,
        "--or",
    );
}

#[test]
fn refus_largeurs_invalides() {
    // chaque largeur < L (sans addition qui déborde), sans doublon (deux largeurs égales = deux flux identiques) — B3
    let c = |l: &str| {
        [
            "intrication",
            "courbe",
            "--l",
            "2",
            "--t",
            "2",
            "--beta",
            "1.0",
            "--largeurs",
            l,
            "--mes",
            "20",
            "--blocs",
            "4",
        ]
        .map(String::from)
    };
    for (nom, l) in [
        ("max", "18446744073709551615"),
        ("egale_l", "2"),
        ("doublon", "1,1"),
    ] {
        let a = c(l);
        let r: Vec<&str> = a.iter().map(String::as_str).collect();
        refus(&format!("largeurs_{nom}"), &r, &[], 2, "--largeurs");
    }
    refus_balayage("l1_sup_l", &[("--l1", "3")], "largeur");
}

/// Ligne JSON du balayage, sans la durée.
fn sans_duree(mut v: serde_json::Value) -> serde_json::Value {
    v.as_object_mut().unwrap().remove("secondes");
    v
}

#[test]
fn courbe_d_une_largeur_egale_le_balayage_de_meme_graine() {
    // la largeur est un CHAMP du flux, pas un ajout à la graine (B1, B3) : la largeur 1 d'une courbe et le balayage 1 → 2 de
    // même graine tirent les mêmes nombres ; vrai aussi pour la graine u64::MAX (aucune arithmétique sur la graine)
    for graine in ["7", "18446744073709551615"] {
        let d = dossier(&format!("egal_{graine}"));
        let (rb, rc) = (d.join("b"), d.join("c"));
        let commun = [
            "--l", "4", "--t", "2", "--beta", "1.0", "--nalpha", "3", "--therm", "5", "--mes",
            "20", "--blocs", "4", "--graine", graine,
        ];
        let mut b = vec![
            "intrication",
            "balayage",
            "--l0",
            "1",
            "--l1",
            "2",
            "--sortie",
            rb.to_str().unwrap(),
        ];
        b.extend(commun);
        let mut c = vec![
            "intrication",
            "courbe",
            "--largeurs",
            "1",
            "--sortie",
            rc.to_str().unwrap(),
        ];
        c.extend(commun);
        let (sb, sc) = (kvu(&b, &[]), kvu(&c, &[]));
        assert_eq!(
            (sb.rc, sc.rc),
            (Some(0), Some(0)),
            "{}\n{}",
            sb.stderr,
            sc.stderr
        );
        let lb: serde_json::Value = serde_json::from_str(
            std::fs::read_to_string(rb.join("resultats.jsonl"))
                .unwrap()
                .trim(),
        )
        .unwrap();
        let lc: serde_json::Value = serde_json::from_str(
            std::fs::read_to_string(rc.join("resultats.jsonl"))
                .unwrap()
                .trim(),
        )
        .unwrap();
        assert_eq!(
            sans_duree(lb),
            sans_duree(lc["balayage"].clone()),
            "graine {graine}"
        );
        let _ = std::fs::remove_dir_all(&d);
    }
}

#[test]
fn memoire_refus_deterministe_avant_ouverture() {
    // admission rendue déterministe par la politique existante (plancher KEVINOTRON_RAM_MIN_GO), sans crochet dans le binaire :
    // refus code 5 avant toute création, quelle que soit la machine
    let a = balayage(&[]);
    let r: Vec<&str> = a.iter().map(String::as_str).collect();
    refus(
        "memoire",
        &r,
        &[("KEVINOTRON_RAM_MIN_GO", "1000000000")],
        5,
        "ressources refusées",
    );
}

#[test]
fn etiquette_experimentale_aide_manifeste_stderr() {
    let d = dossier("statut");
    // aide : commence par [EXPÉRIMENTAL] ; `kvu --help` liste la sous-commande
    let s = kvu(&["intrication", "--help"], &[]);
    assert_eq!(s.rc, Some(0), "{}", s.stderr);
    assert!(
        s.stdout.trim_start().starts_with("[EXPÉRIMENTAL]"),
        "{}",
        s.stdout
    );
    let s = kvu(&["--help"], &[]);
    assert!(
        s.stdout
            .lines()
            .any(|l| l.trim_start().starts_with("intrication") && l.contains("[EXPÉRIMENTAL]")),
        "{}",
        s.stdout
    );
    // run minimal : manifeste `parametres.statut = "experimental"`, ligne d'avertissement EXACTE (mêmes mots que le module tube,
    // sans préfixe) sur stderr, COMPLETE en dernier
    let run = d.join("run");
    let mut a = balayage(&[]);
    a.push(format!("--sortie={}", run.display()));
    let r: Vec<&str> = a.iter().map(String::as_str).collect();
    let s = kvu(&r, &[]);
    assert_eq!(s.rc, Some(0), "{}", s.stderr);
    assert!(
        s.stderr.lines().any(|l| l == AVERTISSEMENT),
        "ligne exacte absente : {}",
        s.stderr
    );
    let m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(run.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(m["parametres"]["statut"], "experimental", "{m}");
    assert_eq!(m["statut"], "termine", "{m}");
    assert!(run.join("COMPLETE").exists() && run.join("resultats.jsonl").exists());
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn avertissement_si_la_largeur_atteint_la_tranche_entiere() {
    // objection 11 : lw + 1 = L ⟹ aire de bord nulle, C sans objet : avertissement sur stderr et dans la ligne de résultat
    let d = dossier("tranche");
    let run = d.join("run");
    let r = run.to_str().unwrap().to_string();
    let a = [
        "intrication",
        "courbe",
        "--l",
        "2",
        "--t",
        "2",
        "--beta",
        "1.0",
        "--largeurs",
        "1",
        "--nalpha",
        "3",
        "--therm",
        "5",
        "--mes",
        "20",
        "--blocs",
        "4",
        "--sortie",
        &r,
    ];
    let s = kvu(&a, &[]);
    assert_eq!(s.rc, Some(0), "{}", s.stderr);
    assert!(s.stderr.contains("aire de bord nulle"), "{}", s.stderr);
    let ligne = std::fs::read_to_string(run.join("resultats.jsonl")).unwrap();
    assert!(ligne.contains("\"avertissement\""), "{ligne}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn les_exemples_de_la_note_tournent() {
    // docs/v0.2/notes/intrication.md : chaque ligne `kvu intrication …` d'un bloc de code est exécutée avec ses arguments,
    // `--sortie X` étant placé sous un répertoire neuf du test ; code 0, manifeste `termine`, COMPLETE
    let note = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/v0.2/notes/intrication.md");
    let texte = std::fs::read_to_string(&note).unwrap();
    let (mut en_bloc, mut cmds) = (false, Vec::new());
    for l in texte.lines() {
        if l.trim_start().starts_with("```") {
            en_bloc = !en_bloc;
        } else if en_bloc && l.trim_start().starts_with("kvu intrication ") {
            cmds.push(l.trim().to_string());
        }
    }
    assert_eq!(
        cmds.len(),
        3,
        "trois exemples attendus dans la note : {cmds:?}"
    );
    let d = dossier("note");
    for c in &cmds {
        let mots: Vec<&str> = c.split_whitespace().skip(1).collect();
        let i = mots
            .iter()
            .position(|m| *m == "--sortie")
            .expect("--sortie dans chaque exemple");
        let rep = d.join(mots[i + 1]);
        let r = rep.to_str().unwrap().to_string();
        let mut args = mots.clone();
        args[i + 1] = &r;
        let s = kvu(&args, &[]);
        assert_eq!(s.rc, Some(0), "« {c} » : {}", s.stderr);
        let m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(rep.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(m["statut"], "termine", "« {c} » : {m}");
        assert!(rep.join("COMPLETE").exists(), "« {c} »");
    }
    let _ = std::fs::remove_dir_all(&d);
}

/// Valeurs HORS du domaine admis, par option (relecture c19b, objection 10 et méthode « un tableau ») : juste dehors de chaque
/// borne, zéro, NaN, infinis, maximum du type. Chacune doit être refusée en code 2, avant tout calcul, sans répertoire créé.
#[test]
fn refus_domaine_parametre() {
    const U64MAX: &str = "18446744073709551615";
    let cas: &[(&str, &[&str])] = &[
        ("--l", &["0", "1", "3", U64MAX]),
        ("--t", &["0", "1", "3", U64MAX]),
        ("--l1", &["3", "0", U64MAX]), // > L, = l0 (ΔF nul par construction), maximum
        (
            "--beta",
            &[
                "0",
                "-0",
                "-1",
                "NaN",
                "inf",
                "-inf",
                "1e-20",
                "0.0009999999999999998",
                "1000.0000000000001",
                "1e308",
                "1.7976931348623157e308",
            ],
        ),
        ("--nalpha", &["0", "1", "2", "4", "32768", "32769", U64MAX]),
        ("--therm", &["4294967276", U64MAX]), // therm + mes × sep = 2³² (mes 20, sep 1)
        ("--mes", &["0", "1", "6", U64MAX]),  // 6 : pas multiple de --blocs 4
        ("--sep", &["0", "214748365", U64MAX]), // 5 + 20 × 214748365 ≥ 2³²
        ("--blocs", &["0", "1", "3", "40", U64MAX]),
        ("--or", &["1001", U64MAX]),
        ("--graine", &["18446744073709551616", "-1"]), // au-delà du type : clap refuse
    ];
    for (opt, valeurs) in cas {
        for v in *valeurs {
            let a = balayage(&[(opt, v)]);
            let r: Vec<&str> = a.iter().map(String::as_str).collect();
            refus(&format!("domaine{opt}{v}"), &r, &[], 2, "");
        }
    }
    let c = |l: &str| {
        [
            "intrication",
            "courbe",
            "--l",
            "4",
            "--t",
            "2",
            "--beta",
            "1.0",
            "--largeurs",
            l,
            "--mes",
            "20",
            "--blocs",
            "4",
        ]
        .map(String::from)
    };
    for l in ["4", "1,1", U64MAX, "0,1,2,3,4"] {
        let a = c(l);
        let r: Vec<&str> = a.iter().map(String::as_str).collect();
        refus(&format!("domaine_largeurs{l}"), &r, &[], 2, "--largeurs");
    }
    let d = |n: &str| {
        [
            "intrication",
            "doublement",
            "--l",
            "2",
            "--t",
            "2",
            "--beta",
            "1.0",
            "--nbeta",
            n,
            "--mes",
            "20",
            "--blocs",
            "4",
        ]
        .map(String::from)
    };
    for n in ["0", "2", "32769", U64MAX] {
        let a = d(n);
        let r: Vec<&str> = a.iter().map(String::as_str).collect();
        refus(&format!("domaine_nbeta{n}"), &r, &[], 2, "--nbeta");
    }
    for (pas, or) in [
        ("0", "0"),
        ("4294967295", "0"),
        (U64MAX, "0"),
        ("1", "1001"),
    ] {
        refus(
            &format!("domaine_debit{pas}_{or}"),
            &[
                "intrication",
                "debit",
                "--l",
                "2",
                "--t",
                "2",
                "--pas",
                pas,
                "--or",
                or,
            ],
            &[],
            2,
            "",
        );
    }
    // capacité des plaquettes (geom::tables : 2·(6V + 3·s·L³) ≤ 2³⁰) : L = T = 82 dehors
    refus(
        "domaine_capacite",
        &["intrication", "debit", "--l", "82", "--t", "82"],
        &[],
        2,
        "",
    );
}

#[test]
fn beta_minuscule_refuse_contre_exemple_c19b() {
    // relecture c19b, bloquante 10 : à β = 1e-20, exp(−2β|A|) vaut 1 en flottant, Creutz rejette tout, les configurations se
    // répètent et l'erreur est nulle ; le domaine admis de β est borné des deux côtés
    let a = [
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
        "1e-20",
        "--nalpha",
        "3",
        "--therm",
        "0",
        "--mes",
        "2",
        "--blocs",
        "2",
        "--or",
        "0",
    ];
    refus("c19b_beta", &a, &[], 2, "--beta");
}

#[test]
fn garde_ssd_de_production_refuse_quand_le_disque_est_plein() {
    // preuve que la garde de production n'est pas affaiblie par l'injection du seuil 0 des autres tests : la MÊME commande
    // valide (run minimal qui calcule et écrit), avec `KEVINOTRON_SSD_MIN_GO=1e9` imposé au sous-processus, est refusée
    // (code 7, `SsdPlein`) AVANT toute écriture ; si `GardeSsd::evaluer` ou `GardeSsd::verifier` est affaibli, ce test rougit
    let d = dossier("garde_ssd");
    let run = d.join("run");
    let mut a = balayage(&[]);
    a.push(format!("--sortie={}", run.display()));
    let r: Vec<&str> = a.iter().map(String::as_str).collect();
    // Aucun `KEVINOTRON_PROFIL` : le profil reste celui du MATÉRIEL (un profil générique différent du matériel ferait
    // remonter le plancher SSD du matériel, et le refus ne serait plus celui de la garde SSD). Le seuil 1e9 Go est imposé
    // par `KEVINOTRON_SSD_MIN_GO`, qui prime sur le profil.
    // `env_avec_seuil_ssd` n'ajoute le seuil 0 des autres tests que si le test n'impose pas lui-même `KEVINOTRON_SSD_MIN_GO` :
    // ici la valeur 1e9 est la seule écrite, donc celle que reçoit le sous-processus.
    let s = kvu(&r, &[(VAR_SEUIL_SSD, "1e9")]);
    assert_eq!(
        s.rc,
        Some(7),
        "garde SSD affaiblie : rc = {:?}\nstdout : {}\nstderr : {}",
        s.rc,
        s.stdout,
        s.stderr
    );
    assert!(
        s.stderr.contains("garde SSD"),
        "stderr sans mention de la garde SSD : {}",
        s.stderr
    );
    assert!(
        !run.join("resultats.jsonl").exists(),
        "resultats.jsonl écrit malgré le refus SSD"
    );
    assert!(
        !run.exists() || std::fs::read_dir(&run).unwrap().next().is_none(),
        "répertoire de run non vide malgré le refus SSD : {:?}",
        std::fs::read_dir(&run).map(|e| e.count())
    );
    let _ = std::fs::remove_dir_all(&d);
}
