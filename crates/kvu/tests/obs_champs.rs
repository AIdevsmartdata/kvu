//! Portes de `kvu obs champs` (module scanner, 11/11/2026) sur une configuration thermalisée L = 4 (β = 2,3, CPU) :
//! sommes contre les observables globales de kvu, localité, invariance de jauge, et un MUTANT par porte (une
//! variante volontairement cassée doit rendre la porte rouge). Test CLI : fichiers, formes, manifeste.

use kvu::commun::ChampHote;
use kvu::heritage::su2_hb::{write_npy_quat, CpuSu2};
use kvu::obs::champs::{
    calculer, jauge_aleatoire, porte_localite, portes_jauge, portes_somme, references,
    ParamsChamps, Porte,
};
use std::path::{Path, PathBuf};

fn config(l: usize, graine: u64, n_pas: usize) -> ChampHote {
    let mut c = CpuSu2::new(l, graine).unwrap();
    c.init(true);
    for _ in 0..n_pas {
        c.step(2.3, 2);
    }
    ChampHote { l, q: c.flat() }
}

const P: ParamsChamps = ParamsChamps {
    dt: 0.01,
    n_pas: 20,
};

fn porte<'a>(v: &'a [Porte], id: &str) -> &'a Porte {
    v.iter()
        .find(|p| p.id == id)
        .unwrap_or_else(|| panic!("porte {id} absente"))
}

#[test]
fn portes_somme_vertes_et_mutants_rouges() {
    let ch = config(4, 11, 30);
    let c = calculer(&ch, &P).unwrap();
    let r = references(&ch, &P).unwrap();
    let ps = portes_somme(&c, &r, &P);
    for p in &ps {
        eprintln!("{}", p.ligne());
        assert!(p.ok, "{}", p.ligne());
    }
    assert!(
        r.negatives_brut > 0.0,
        "configuration chaude : des plaquettes négatives attendues"
    );

    // MUTANT 1 : action sans le plan 23 (une plaquette oubliée) -> porte de somme rouge.
    let mut m = c.clone();
    let res = kvu::obs::ref_cpu::Reseau::nouveau(&ch).unwrap();
    for s in 0..m.action_brute.len() {
        let t = kvu::obs::champs::traces_plaquettes(&res, s);
        m.action_brute[s] = 4.0 * t[..5].iter().map(|x| 1.0 - x).sum::<f64>();
    }
    assert!(!porte(&portes_somme(&m, &r, &P), "CH-SOMME-ACTION-BRUTE").ok);
    // MUTANT 2 : charge de signe opposé -> rouge.
    let mut m = c.clone();
    m.charge_flot.iter_mut().for_each(|x| *x = -*x);
    assert!(!porte(&portes_somme(&m, &r, &P), "CH-SOMME-CHARGE-FLOT").ok);
    // MUTANT 3 : un pas de flot de trop dans les champs -> rouge (action et trèfle).
    let p2 = ParamsChamps {
        dt: 0.01,
        n_pas: 21,
    };
    let c2 = calculer(&ch, &p2).unwrap();
    let ps2 = portes_somme(&c2, &r, &P);
    assert!(!porte(&ps2, "CH-SOMME-ACTION-FLOT").ok);
    assert!(!porte(&ps2, "CH-SOMME-TREFLE-FLOT").ok);
    // MUTANT 4 : un site de plus compté négatif -> rouge (égalité exacte).
    let mut m = c.clone();
    m.plaq_negatives_brute[0] += 1.0;
    assert!(!porte(&portes_somme(&m, &r, &P), "CH-SOMME-NEGATIVES-BRUTE").ok);
    // MUTANT 5 : Polyakov de la direction 0 mis à l'échelle 1,001 -> rouge.
    let mut m = c.clone();
    let v = 4usize.pow(4);
    m.polyakov[..v].iter_mut().for_each(|x| *x *= 1.001);
    assert!(!porte(&portes_somme(&m, &r, &P), "CH-SOMME-POLYAKOV-0").ok);
}

#[test]
fn porte_localite_verte_et_mutant_rouge() {
    let ch = config(4, 12, 20);
    let c = calculer(&ch, &P).unwrap();
    let p = porte_localite(&ch, &c, 64).unwrap();
    assert!(p.ok, "{}", p.ligne());
    // MUTANT : champ d'action translaté d'un site (même somme : les portes de somme restent vertes) -> rouge.
    let mut m = c.clone();
    m.action_brute.rotate_left(1);
    assert!(!porte_localite(&ch, &m, 64).unwrap().ok);
    let r = references(&ch, &P).unwrap();
    assert!(
        porte(&portes_somme(&m, &r, &P), "CH-SOMME-ACTION-BRUTE").ok,
        "la somme ne voit pas une permutation"
    );
}

#[test]
fn porte_jauge_verte_et_mutant_rouge() {
    let ch = config(4, 13, 20);
    let c = calculer(&ch, &P).unwrap();
    let g = jauge_aleatoire(&ch, 5);
    assert!(
        (g.q[0] - ch.q[0]).abs() > 1e-3,
        "la transformation de jauge doit changer les liens"
    );
    let cg = calculer(&g, &P).unwrap();
    for p in portes_jauge(&c, &cg) {
        eprintln!("{}", p.ligne());
        assert!(p.ok, "{}", p.ligne());
    }
    // MUTANT : un « champ » non invariant (½ Tr U_0(x), dépend de la jauge) à la place de l'action -> rouge.
    let v = 4usize.pow(4);
    let lien0 = |h: &ChampHote| -> Vec<f64> { (0..v).map(|s| h.q[4 * s]).collect() };
    let (mut a, mut b) = (c.clone(), cg.clone());
    a.action_brute = lien0(&ch);
    b.action_brute = lien0(&g);
    assert!(!porte(&portes_jauge(&a, &b), "CH-JAUGE-ACTION_BRUTE").ok);
}

#[test]
fn champ_froid_nul() {
    let ch = ChampHote::froid(4).unwrap();
    let c = calculer(&ch, &P).unwrap();
    assert!(c.action_brute.iter().all(|&x| x.abs() < 1e-14));
    assert!(c.charge_flot.iter().all(|&x| x.abs() < 1e-14));
    assert!(c.polyakov.iter().all(|&x| (x - 1.0).abs() < 1e-14));
    assert!(c.plaq_negatives_brute.iter().all(|&x| x == 0.0));
}

fn dossier_test(nom: &str) -> PathBuf {
    let base = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let d = base.join(format!("kvu_obs_champs_{nom}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Lance `kvu obs <args>` dans un SOUS-PROCESSUS isolé (comme `tests/obs_topo_cli.rs`) : variables KEVINOTRON_* / KVU_*
/// héritées retirées, seuil SSD de test explicite (les essais ne dépendent pas de la politique disque de la machine),
/// verrou GPU ignoré, répertoire courant `cwd` (chemins relatifs). Rend (code de retour, stdout, stderr).
fn lancer_dans(cwd: &Path, args: &[&str]) -> (i32, String, String) {
    let mut c = std::process::Command::new(env!("CARGO_BIN_EXE_kvu"));
    for (k, _) in
        std::env::vars().filter(|(k, _)| k.starts_with("KEVINOTRON_") || k.starts_with("KVU_"))
    {
        c.env_remove(k);
    }
    let o = c
        .env("KEVINOTRON_VERROU", "ignorer")
        .env("KEVINOTRON_RAM_MIN_GO", "0")
        .env("KEVINOTRON_SSD_MIN_GO", "0")
        .env("RAYON_NUM_THREADS", "4")
        .current_dir(cwd)
        .arg("obs")
        .args(args)
        .output()
        .unwrap();
    (
        o.status.code().expect("terminaison par signal"),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

fn lancer(args: &[&str]) -> i32 {
    lancer_dans(Path::new(env!("CARGO_TARGET_TMPDIR")), args).0
}

/// Code de retour de `Erreur::Config` (usage).
const RC_USAGE: i32 = 2;

fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// Relit un NPY v1.0 : (en-tête, octets de données).
fn lire_npy(chemin: &Path) -> (String, Vec<u8>) {
    let o = std::fs::read(chemin).unwrap();
    assert_eq!(&o[..6], b"\x93NUMPY");
    let n = u16::from_le_bytes([o[8], o[9]]) as usize;
    (
        std::str::from_utf8(&o[10..10 + n]).unwrap().to_string(),
        o[10 + n..].to_vec(),
    )
}

#[test]
fn cli_ecrit_les_champs_et_le_manifeste() {
    let d = dossier_test("cli");
    let cfg = d.join("cfg_000001.npy");
    let ch = config(4, 14, 20);
    write_npy_quat(&cfg, 4, &ch.q).unwrap();
    let out = d.join("export");
    let rc = lancer(&[
        "champs",
        "--cfg",
        &s(&cfg),
        "--out",
        &s(&out),
        "--t-flot",
        "0.2",
        "--beta",
        "2.3",
        "--porte-jauge",
        "--f64",
    ]);
    assert_eq!(rc, 0);
    // manifeste COMMUN (stockage::provenance::Manifeste) : COMPLETE, fichiers consignés, geometrie, kvu_champs
    assert!(out.join("COMPLETE").is_file());
    assert!(!out.join("manifeste.json").exists());
    let top: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(top["statut"], "termine");
    assert_eq!(top["rc"], 0);
    assert_eq!(top["fichiers"].as_array().unwrap().len(), 7);
    // O5 : le calcul est toujours CPU FP64, quel que soit le backend que la détection de la machine aurait choisi
    assert_eq!(top["backend"], "cpu");
    assert!(
        top["arch"].is_null(),
        "pas d'architecture GPU pour un calcul CPU : {}",
        top["arch"]
    );
    assert_eq!(top["precision_calcul"], "FP64");
    assert_eq!(top["geometrie"]["direction_temps"], 3);
    assert_eq!(
        top["geometrie"]["dimensions"],
        serde_json::json!([4, 4, 4, 4])
    );
    let man = &top["kvu_champs"];
    assert_eq!(man["format"], "kvu-champs/1");
    assert_eq!(man["porte_jauge"]["graine"], 1);
    assert_eq!(man["l"], 4);
    assert_eq!(man["flot"]["n_pas"], 20);
    assert_eq!(man["portes_ok"], true);
    let n_portes = man["portes"].as_array().unwrap().len();
    assert_eq!(
        n_portes,
        11 + 1 + 7,
        "11 portes de somme, 1 de localité, 7 de jauge"
    );
    // relecture f64 : la somme du fichier = somme FP64 du manifeste (au bit, même ordre de réduction non garanti : 1e-12)
    let (entete, octets) = lire_npy(&out.join("action_brute.npy"));
    assert!(
        entete.contains("'<f8'") && entete.contains("(4, 4, 4, 4)"),
        "{entete}"
    );
    let x: Vec<f64> = octets
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let somme: f64 = x.iter().sum();
    let attendu = man["champs"]["action_brute"]["somme_f64"].as_f64().unwrap();
    assert!((somme - attendu).abs() <= 1e-12 * attendu.abs());
    let (entete, _) = lire_npy(&out.join("polyakov.npy"));
    assert!(entete.contains("(4, 4, 4, 4, 4)"), "{entete}");
    // f32 par défaut
    let out32 = d.join("export32");
    assert_eq!(
        lancer(&[
            "champs",
            "--cfg",
            &s(&cfg),
            "--out",
            &s(&out32),
            "--n-pas",
            "5"
        ]),
        0
    );
    let (entete, octets) = lire_npy(&out32.join("charge_flot.npy"));
    assert!(entete.contains("'<f4'"), "{entete}");
    assert_eq!(octets.len(), 4 * 256);
    // un répertoire de sortie existant est refusé (manifeste commun : répertoire neuf)
    assert!(
        lancer(&[
            "champs",
            "--cfg",
            &s(&cfg),
            "--out",
            &s(&out32),
            "--n-pas",
            "5"
        ]) != 0
    );
    // usage : --t-flot et --n-pas ensemble refusés
    assert!(
        lancer(&[
            "champs",
            "--cfg",
            &s(&cfg),
            "--out",
            &s(&out32),
            "--n-pas",
            "5",
            "--t-flot",
            "0.05"
        ]) != 0
    );
}

/// Refus des configurations tordues ou non périodiques (les densités exportées ignorent les z_p = −1 de la torsion) :
/// rien n'est créé. Une torsion triviale déclarée et un manifeste su2-gen sans bord sont acceptés.
#[test]
fn refus_configuration_tordue_ou_a_bord() {
    let ch = config(4, 15, 5);
    let cas: [(&str, &str, &str, bool); 6] = [
        (
            "torsion_json",
            "torsion.json",
            r#"{"plans": ["01"], "pile": "coins (L-1, L-1)", "facteur": -1}"#,
            false,
        ),
        (
            "manifest_torsion",
            "manifest.json",
            r#"{"mode": "su2-gen", "torsion": {"plans": ["01", "23"]}}"#,
            false,
        ),
        (
            "manifest_bord",
            "manifest.json",
            r#"{"mode": "su2-bord", "bord": [{"chaine": 0}]}"#,
            false,
        ),
        (
            "config_run_bord",
            "config_run.json",
            r#"{"bord": {"mode": "x0"}}"#,
            false,
        ),
        ("torsion_triviale", "torsion.json", r#"{"plans": []}"#, true),
        (
            "manifest_periodique",
            "manifest.json",
            r#"{"mode": "su2-gen", "config_run": {"bord": null}}"#,
            true,
        ),
    ];
    for (nom, fichier, contenu, accepte) in cas {
        let d = dossier_test(&format!("refus_{nom}"));
        let cfg = d.join("cfg_000001.npy");
        write_npy_quat(&cfg, 4, &ch.q).unwrap();
        std::fs::write(d.join(fichier), contenu).unwrap();
        let out = d.join("export");
        let r = lancer(&[
            "champs",
            "--cfg",
            &s(&cfg),
            "--out",
            &s(&out),
            "--n-pas",
            "2",
            "--sans-portes",
        ]);
        if accepte {
            assert_eq!(r, 0, "{nom} doit être accepté");
            let top: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap())
                    .unwrap();
            assert!(
                !top["geometrie"]["declaration"]["fichiers_lus"]
                    .as_array()
                    .unwrap()
                    .is_empty(),
                "{nom}"
            );
        } else {
            assert_eq!(r, RC_USAGE, "{nom} : refus de configuration attendu");
            assert!(!out.exists(), "{nom} : aucun répertoire ne doit être créé");
        }
    }
    // configuration tordue déclarée dans le dossier PARENT (dumps rangés en c000/) : refusée aussi
    let d = dossier_test("refus_parent");
    std::fs::create_dir_all(d.join("c000")).unwrap();
    let cfg = d.join("c000").join("cfg_000001.npy");
    write_npy_quat(&cfg, 4, &ch.q).unwrap();
    std::fs::write(d.join("torsion.json"), r#"{"plans": ["12"]}"#).unwrap();
    assert_eq!(
        lancer(&[
            "champs",
            "--cfg",
            &s(&cfg),
            "--out",
            &s(&d.join("export")),
            "--n-pas",
            "2",
            "--sans-portes"
        ]),
        RC_USAGE
    );
    // sans aucune métadonnée : supposée périodique, et c'est consigné
    let d = dossier_test("sans_meta");
    let cfg = d.join("cfg_000001.npy");
    write_npy_quat(&cfg, 4, &ch.q).unwrap();
    let out = d.join("export");
    assert_eq!(
        lancer(&[
            "champs",
            "--cfg",
            &s(&cfg),
            "--out",
            &s(&out),
            "--n-pas",
            "2",
            "--sans-portes"
        ]),
        0
    );
    let top: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    assert!(top["geometrie"]["declaration"]
        .as_str()
        .unwrap()
        .contains("SUPPOSÉE périodique"));
}

/// O1 (relecture 11/11) : le refus ne dépend pas de la façon d'écrire le chemin de la configuration. Torsion ou bord déclarés
/// à la racine d'un lot, configuration dans `c000/` : refus pour le chemin absolu, `c000/cfg…` depuis la racine, le fichier
/// nu depuis `c000/`, un chemin avec `..`, et un lien symbolique placé ailleurs.
#[test]
fn refus_independant_de_la_forme_du_chemin() {
    let ch = config(4, 16, 5);
    for (fichier, contenu) in [
        ("torsion.json", r#"{"plans": ["01"]}"#),
        (
            "manifest.json",
            r#"{"mode": "su2-bord", "bord": [{"chaine": 0}]}"#,
        ),
    ] {
        let d = dossier_test(&format!("chemins_{}", fichier.replace('.', "_")));
        std::fs::create_dir_all(d.join("c000")).unwrap();
        let abs = d.join("c000").join("cfg_000001.npy");
        write_npy_quat(&abs, 4, &ch.q).unwrap();
        std::fs::write(d.join(fichier), contenu).unwrap();
        let ailleurs = d.join("ailleurs");
        std::fs::create_dir_all(&ailleurs).unwrap();
        std::os::unix::fs::symlink(&abs, ailleurs.join("lien.npy")).unwrap();
        let cas: [(PathBuf, &str); 5] = [
            (d.clone(), "c000/cfg_000001.npy"),
            (d.join("c000"), "cfg_000001.npy"),
            (d.join("c000"), "./cfg_000001.npy"),
            (d.join("c000"), "../c000/cfg_000001.npy"),
            (ailleurs.clone(), "lien.npy"),
        ];
        for (k, (cwd, cfg)) in cas.iter().enumerate() {
            let out = d.join(format!("export{k}"));
            let (rc, _, err) = lancer_dans(
                cwd,
                &[
                    "champs",
                    "--cfg",
                    cfg,
                    "--out",
                    &s(&out),
                    "--n-pas",
                    "2",
                    "--sans-portes",
                ],
            );
            assert_eq!(
                rc,
                RC_USAGE,
                "{fichier} : cwd={} cfg={cfg} doit être refusé ({err})",
                cwd.display()
            );
            assert!(
                !out.exists(),
                "{fichier} : cwd={} cfg={cfg} : aucun répertoire ne doit être créé",
                cwd.display()
            );
        }
        // chemin absolu : toujours refusé
        let out = d.join("export_abs");
        assert_eq!(
            lancer(&[
                "champs",
                "--cfg",
                &s(&abs),
                "--out",
                &s(&out),
                "--n-pas",
                "2",
                "--sans-portes"
            ]),
            RC_USAGE
        );
    }
    // même dossier, chemin nu, métadonnée voisine périodique : accepté (le correctif ne refuse pas à tort)
    let d = dossier_test("chemins_ok");
    let abs = d.join("cfg_000001.npy");
    write_npy_quat(&abs, 4, &ch.q).unwrap();
    std::fs::write(d.join("torsion.json"), r#"{"plans": []}"#).unwrap();
    let out = d.join("export");
    let (rc, _, err) = lancer_dans(
        &d,
        &[
            "champs",
            "--cfg",
            "cfg_000001.npy",
            "--out",
            &s(&out),
            "--n-pas",
            "2",
            "--sans-portes",
        ],
    );
    assert_eq!(rc, 0, "{err}");
}

/// O2 (relecture 11/11) : `f64::max` ignore les NaN, donc des champs non finis pouvaient passer les portes de maximum
/// (localité, jauge). Mutants NaN et infinis : toutes ces portes doivent être ROUGES ; et l'export refuse un champ non fini
/// même sans portes.
#[test]
fn champs_non_finis_portes_rouges_et_export_refuse() {
    let ch = config(4, 17, 20);
    let c = calculer(&ch, &P).unwrap();
    let g = calculer(&jauge_aleatoire(&ch, 5), &P).unwrap();
    for (nom, mauvais) in [
        ("NaN", f64::NAN),
        ("+inf", f64::INFINITY),
        ("-inf", f64::NEG_INFINITY),
    ] {
        // localité : champ d'action non fini en tous les sites (y compris ceux qui ne sont pas tirés)
        let mut m = c.clone();
        m.action_brute.iter_mut().for_each(|x| *x = mauvais);
        assert!(
            !porte_localite(&ch, &m, 64).unwrap().ok,
            "localité {nom} (action)"
        );
        let mut m = c.clone();
        m.polyakov.iter_mut().for_each(|x| *x = mauvais);
        assert!(
            !porte_localite(&ch, &m, 64).unwrap().ok,
            "localité {nom} (polyakov)"
        );
        // jauge : les deux côtés non finis, ou un seul
        let mut a = c.clone();
        let mut b = g.clone();
        a.charge_flot.iter_mut().for_each(|x| *x = mauvais);
        b.charge_flot.iter_mut().for_each(|x| *x = mauvais);
        assert!(
            !porte(&portes_jauge(&a, &b), "CH-JAUGE-CHARGE_FLOT").ok,
            "jauge {nom} des deux côtés"
        );
        let mut b = g.clone();
        b.energie_trefle_flot[3] = mauvais;
        assert!(
            !porte(&portes_jauge(&c, &b), "CH-JAUGE-ENERGIE_TREFLE_FLOT").ok,
            "jauge {nom} d'un côté"
        );
        let mut a = c.clone();
        a.action_flot[7] = mauvais;
        assert!(
            !porte(&portes_jauge(&a, &g), "CH-JAUGE-ACTION_FLOT").ok,
            "jauge {nom} côté référence"
        );
    }
    // CLI : un pas de flot énorme (fini, accepté par les paramètres) sature la norme des liens ; l'export est refusé
    // et aucun COMPLETE n'est écrit, avec ou sans portes.
    let d = dossier_test("non_finis");
    let cfg = d.join("cfg_000001.npy");
    write_npy_quat(&cfg, 4, &ch.q).unwrap();
    for (k, extra) in [
        &["--sans-portes", "--porte-jauge"][..],
        &[][..],
        &["--sans-portes"][..],
    ]
    .iter()
    .enumerate()
    {
        let out = d.join(format!("export{k}"));
        let mut args = vec![
            "champs",
            "--cfg",
            s_(&cfg),
            "--out",
            s_(&out),
            "--n-pas",
            "3",
            "--dt",
            "1e200",
        ];
        args.extend_from_slice(extra);
        let (rc, _, err) = lancer_dans(&d, &args);
        assert_ne!(rc, 0, "dt énorme : l'export non fini doit échouer ({err})");
        assert!(
            !out.join("COMPLETE").exists(),
            "dt énorme : pas de COMPLETE"
        );
    }
}

fn s_(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// O3 (relecture 11/11) : `--t-flot 1 --dt 1e-30` donnait ~1e30 pas, saturés en silence à `usize::MAX` par `as usize`.
/// Un nombre de pas non représentable (ou un temps dérivé non fini) est refusé AVANT tout calcul, rien n'est créé.
#[test]
fn nombre_de_pas_non_representable_refuse() {
    let ch = config(4, 18, 5);
    let d = dossier_test("pas");
    let cfg = d.join("cfg_000001.npy");
    write_npy_quat(&cfg, 4, &ch.q).unwrap();
    let cas: [&[&str]; 5] = [
        &["--t-flot", "1", "--dt", "1e-30"],
        &["--t-flot", "1e300", "--dt", "1"],
        &["--n-pas", "5", "--dt", "1e308"],
        &["--t-flot", "inf"],
        &["--t-flot", "1", "--dt", "1e-17"],
    ];
    for (k, extra) in cas.iter().enumerate() {
        let out = d.join(format!("export{k}"));
        let mut args = vec![
            "champs",
            "--cfg",
            s_(&cfg),
            "--out",
            s_(&out),
            "--sans-portes",
        ];
        args.extend_from_slice(extra);
        let (rc, _, err) = lancer_dans(&d, &args);
        assert_eq!(rc, RC_USAGE, "{extra:?} doit être refusé en usage ({err})");
        assert!(
            !out.exists(),
            "{extra:?} : aucun répertoire ne doit être créé"
        );
    }
    // cas valide voisin : accepté
    let out = d.join("export_ok");
    assert_eq!(
        lancer(&[
            "champs",
            "--cfg",
            s_(&cfg),
            "--out",
            s_(&out),
            "--t-flot",
            "0.04",
            "--sans-portes"
        ]),
        0
    );
}

/// O4 (relecture 11/11) : sur un champ nul ou presque nul, le dénominateur max|f| de la porte de jauge est ~0 et la porte
/// exigeait un écart absolu < 1e-310 alors qu'une transformation de jauge introduit des arrondis de 1e-16. Elle doit être
/// VERTE sur la configuration froide et presque froide, et rester ROUGE pour un champ réellement non invariant.
#[test]
fn porte_jauge_utilisable_sur_champ_froid_et_presque_froid() {
    let froid = ChampHote::froid(4).unwrap();
    let mut presque = froid.clone();
    // un lien tourné de 1e-4 rad : champs d'ordre 1e-8, loin de l'arrondi mais bien en dessous de 1
    let e = 1e-4f64;
    presque.q[0] = e.cos();
    presque.q[1] = e.sin();
    for (nom, ch) in [("froid", &froid), ("presque_froid", &presque)] {
        let c = calculer(ch, &P).unwrap();
        for graine in [1, 5, 9] {
            let cg = calculer(&jauge_aleatoire(ch, graine), &P).unwrap();
            for p in portes_jauge(&c, &cg) {
                assert!(p.ok, "{nom} graine {graine} : {}", p.ligne());
            }
        }
        // mutant : un « champ » dépendant de la jauge (½ Tr U_0) à la place de l'action reste rouge, même sur champ nul
        let g = jauge_aleatoire(ch, 5);
        let v = 4usize.pow(4);
        let (mut a, mut b) = (c.clone(), c.clone());
        a.action_brute = (0..v).map(|s| ch.q[4 * s]).collect();
        b.action_brute = (0..v).map(|s| g.q[4 * s]).collect();
        assert!(
            !porte(&portes_jauge(&a, &b), "CH-JAUGE-ACTION_BRUTE").ok,
            "{nom} : mutant non invariant"
        );
    }
    // CLI de bout en bout sur la configuration froide, porte de jauge jouée
    let d = dossier_test("froid_jauge");
    let cfg = d.join("cfg_000001.npy");
    write_npy_quat(&cfg, 4, &froid.q).unwrap();
    let out = d.join("export");
    let (rc, o, err) = lancer_dans(
        &d,
        &[
            "champs",
            "--cfg",
            s_(&cfg),
            "--out",
            s_(&out),
            "--n-pas",
            "5",
            "--porte-jauge",
        ],
    );
    assert_eq!(rc, 0, "{o}\n{err}");
}
