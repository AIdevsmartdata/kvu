//! Contrat CLI topologique : uniquement CPU, subprocessus isolés, aucune ouverture GPU.

use kvu::heritage::manifest::sha256_fichier;
use kvu::obs::topo::{FlotA2, RecordTopo};
use kvu::obs::topo_gates::{analyser, LigneGate};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::Instant;

struct Dossier(PathBuf);

impl Dossier {
    fn nouveau(nom: &str) -> Self {
        let p = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("obs_topo_cli_{nom}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        Self(p)
    }

    fn commande(&self) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_kvu"));
        for (k, _) in
            std::env::vars().filter(|(k, _)| k.starts_with("KEVINOTRON_") || k.starts_with("KVU_"))
        {
            c.env_remove(k);
        }
        c.env("KEVINOTRON_VERROU", "ignorer")
            .env("KEVINOTRON_RAM_MIN_GO", "0")
            .env("KEVINOTRON_SSD_MIN_GO", "0")
            .env("KEVINOTRON_CACHE_DIR", self.0.join("cache"))
            .env("KEVINOTRON_STORE", self.0.join("store"))
            .env("KEVINOTRON_FILE", self.0.join("file"))
            .env("TMPDIR", &self.0)
            .env("RAYON_NUM_THREADS", "4");
        c
    }

    fn kvu(&self, args: &[&str]) -> Output {
        self.commande().args(args).output().unwrap()
    }
}

impl Drop for Dossier {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}
fn texte(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).unwrap()
}
fn rc(o: &Output, attendu: i32) {
    assert_eq!(
        o.status.code(),
        Some(attendu),
        "stdout={}\nstderr={}",
        texte(o),
        String::from_utf8_lossy(&o.stderr)
    );
}
fn records(p: &Path) -> Vec<RecordTopo> {
    fs::read_to_string(p)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect()
}
fn json(p: &Path) -> Value {
    serde_json::from_slice(&fs::read(p).unwrap()).unwrap()
}

#[test]
fn ensemble_cpu_modes_reprise_et_provenance() {
    let d = Dossier::nouveau("ensemble");
    let ensemble = d.0.join("ensemble");
    rc(
        &d.kvu(&[
            "su2",
            "gen",
            "--backend",
            "cpu",
            "--L",
            "4",
            "--beta",
            "2.3",
            "--graine",
            "26102619010",
            "--start",
            "hot",
            "--therm",
            "2",
            "--nmeas",
            "6",
            "--skip",
            "1",
            "--nor",
            "0",
            "--dump-every",
            "1",
            "--outdir",
            s(&ensemble),
        ]),
        0,
    );
    let complet = d.0.join("complet");
    let partiel = d.0.join("partiel");
    let mesure = |out: &Path, supplement: &[&str]| {
        let mut args = vec![
            "obs",
            "topo",
            "--ensemble",
            s(&ensemble),
            "--out",
            s(out),
            "--ens",
            "lit",
            "--chaine",
            "2",
            "--backend-gen",
            "hip",
            "--backend",
            "cpu",
            "--pas-max",
            "500",
        ];
        args.extend_from_slice(supplement);
        d.kvu(&args)
    };
    let o = mesure(&complet, &[]);
    rc(&o, 0);
    assert_eq!(texte(&o).lines().count(), 6);
    let rs = records(&complet.join("topo.jsonl"));
    assert_eq!(rs.len(), 6);
    for (i, r) in rs.iter().enumerate() {
        assert_eq!(r.cfg, i + 1);
        assert_eq!(
            (
                &r.ens[..],
                r.chaine,
                &r.backend_gen[..],
                &r.backend_mesure[..],
                r.l
            ),
            ("lit", 2, "hip", "cpu", 4)
        );
        assert_eq!(
            r.flot_a2.q_imp.len(),
            4,
            "défaut toujours complet, même pour lit"
        );
        assert_eq!(r.cooling.q_imp.len(), 21);
        assert_eq!(
            r.fichier,
            s(&ensemble.join(format!("cfg_{:06}.npy", i + 1)))
        );
    }
    // Un sous-niveau c000/ est toléré ; il porte toujours une seule chaîne.
    fs::create_dir(ensemble.join("c000")).unwrap();
    for cfg in 1..=6 {
        let nom = format!("cfg_{cfg:06}.npy");
        fs::rename(ensemble.join(&nom), ensemble.join("c000").join(&nom)).unwrap();
    }
    rc(&mesure(&partiel, &["--flot-complet-chaque", "3"]), 0);
    let p = partiel.join("topo.jsonl");
    let original = fs::read(&p).unwrap();
    for r in records(&p) {
        let n = if r.cfg % 3 == 0 { 4 } else { 3 };
        assert_eq!(r.flot_a2.q_imp.len(), n);
        assert_eq!(r.flot_a2.q_tre.len(), n);
        assert_eq!(r.flot_a2.e_plaq.len(), n);
        if n == 3 {
            assert_eq!(r.t0, None);
            assert_eq!(r.flot_t0, None);
        }
        if let Some(t0) = &r.flot_t0 {
            assert!(r.t0.is_some());
            assert_eq!(t0.q_imp.len(), 4);
            assert_eq!(t0.q_tre.len(), 4);
            assert_eq!(t0.t_eff.len(), 4);
        }
    }
    let refus = mesure(&partiel, &["--flot-complet-chaque", "3"]);
    assert_ne!(refus.status.code(), Some(0));
    assert_eq!(fs::read(&p).unwrap(), original);
    let manifeste_initial = fs::read(partiel.join("manifest.json")).unwrap();
    let texte_original = std::str::from_utf8(&original).unwrap();
    let prefixe: String = texte_original
        .lines()
        .take(3)
        .map(|l| format!("{l}\n"))
        .collect();
    fs::write(&p, &prefixe).unwrap();
    rc(
        &mesure(&partiel, &["--flot-complet-chaque", "3", "--reprendre"]),
        0,
    );
    assert_eq!(
        fs::read(&p).unwrap(),
        original,
        "reprise identique à l'octet"
    );
    assert_eq!(
        fs::read(partiel.join("manifest.json")).unwrap(),
        manifeste_initial
    );
    let man = json(&partiel.join("reprises/r000001/manifest.json"));
    assert_eq!(man["backend"], "cpu");
    assert_eq!(man["verrou"], "ignore");
    assert_eq!(man["entrees"].as_array().unwrap().len(), 6);
    assert!(man["argv"]
        .as_array()
        .unwrap()
        .iter()
        .any(|v| v == "--reprendre"));
    let hash = sha256_fichier(&p).unwrap();
    assert!(man["fichiers"]
        .as_array()
        .unwrap()
        .iter()
        .any(|f| f["sha256"] == hash));
    for entree in man["entrees"].as_array().unwrap() {
        assert_eq!(
            entree["sha256"],
            sha256_fichier(Path::new(entree["chemin"].as_str().unwrap())).unwrap()
        );
    }
    let modifies = records(&p);
    for (champ, valeur) in [
        ("ens", Value::from("autre")),
        ("chaine", Value::from(99)),
        ("backend_gen", Value::from("cuda")),
    ] {
        let mut r = serde_json::to_value(&modifies[0]).unwrap();
        r[champ] = valeur;
        let incompatible = format!("{r}\n");
        fs::write(&p, &incompatible).unwrap();
        let o = mesure(&partiel, &["--reprendre"]);
        rc(&o, 2);
        assert!(String::from_utf8_lossy(&o.stderr).contains("incompatible"));
        assert_eq!(fs::read_to_string(&p).unwrap(), incompatible);
    }
    // Dernière ligne interrompue : seules les lignes complètes sont conservées.
    let mut interrompu = format!("{prefixe}{{\"ens\":\"lit\"").into_bytes();
    interrompu.push(0xc3); // Premier octet d'un caractère UTF-8 interrompu.
    fs::write(&p, interrompu).unwrap();
    rc(
        &mesure(&partiel, &["--flot-complet-chaque", "3", "--reprendre"]),
        0,
    );
    assert_eq!(fs::read(&p).unwrap(), original);
    // Un dernier JSON complet sans LF est conservé et séparé de la suite.
    fs::write(&p, texte_original.trim_end_matches('\n')).unwrap();
    rc(&mesure(&partiel, &["--reprendre"]), 0);
    assert_eq!(fs::read(&p).unwrap(), original);
    // Une ligne invalide terminée par LF n'est jamais éliminée silencieusement.
    fs::write(&p, "{}\n").unwrap();
    rc(&mesure(&partiel, &["--reprendre"]), 4);
    assert_eq!(fs::read_to_string(&p).unwrap(), "{}\n");
    rc(
        &mesure(&d.0.join("invalide"), &["--flot-complet-chaque", "0"]),
        2,
    );
}

#[test]
fn inst_cpu_une_ligne_et_valeurs_scientifiques() {
    let d = Dossier::nouveau("inst");
    let debut = Instant::now();
    let o = d.kvu(&["obs", "topo-gates", "inst", "--backend", "cpu"]);
    rc(&o, 0);
    let t = texte(&o);
    assert_eq!(t.lines().count(), 1);
    assert!(t.starts_with("PASS TOPO-INST backend=cpu"), "{t}");
    for l in [8, 12, 16] {
        assert!(t.contains(&format!("L={l} ")));
    }
    assert!(t.contains("gel_imp=") && t.contains("jauge_imp=") && t.contains("reflexion_imp="));
    eprintln!("TOPO-INST CPU : {:.3}s", debut.elapsed().as_secs_f64());
}

#[test]
fn cool_id_cpu_et_mutant() {
    let d = Dossier::nouveau("cool");
    let o = d.kvu(&["obs", "topo-gates", "cool-id", "--backend", "cpu"]);
    rc(&o, 0);
    let t = texte(&o);
    let lignes: Vec<_> = t.lines().collect();
    assert_eq!(lignes.len(), 2);
    assert!(lignes[0].starts_with("PASS TOPO-COOL-ID backend=cpu"));
    assert!(lignes[1].starts_with("PASS TOPO-COOL-ID-MUT backend=cpu"));
}

#[test]
fn analyse_obs4_concatene_et_copie_stdout() {
    let d = Dossier::nouveau("analyse");
    let records: Vec<_> = (1..=20)
        .map(|cfg| RecordTopo {
            ens: "obs4".into(),
            cfg,
            chaine: 1,
            l: 16,
            flot_a2: FlotA2 {
                q_imp: vec![0.1, 0.1, cfg as f64 * 0.1, 0.1],
                ..FlotA2::default()
            },
            ..RecordTopo::default()
        })
        .collect();
    let a = d.0.join("a.jsonl");
    let b = d.0.join("b.jsonl");
    for (p, rs) in [(&a, &records[..10]), (&b, &records[10..])] {
        let texte: String = rs
            .iter()
            .map(|r| format!("{}\n", serde_json::to_string(r).unwrap()))
            .collect();
        fs::write(p, texte).unwrap();
    }
    let out = d.0.join("lignes.txt");
    let liste = format!("{},{}", a.display(), b.display());
    let o = d.kvu(&[
        "obs",
        "topo-gates",
        "analyse",
        "--ens",
        "obs4",
        "--jsonl",
        &liste,
        "--lignes",
        s(&out),
    ]);
    rc(&o, 0);
    let attendu: String = analyser("obs4", &records)
        .iter()
        .map(|l| format!("{l}\n"))
        .collect();
    assert_eq!(texte(&o), attendu);
    assert!(attendu.starts_with("INFO OBS-4-L16-REJEU "));
    assert_eq!(fs::read_to_string(&out).unwrap(), attendu);
    let o = d.kvu(&[
        "obs",
        "topo-gates",
        "analyse",
        "--ens",
        "q",
        "--jsonl",
        &liste,
    ]);
    let attendu: String = analyser("q", &records)
        .iter()
        .map(|l| format!("{l}\n"))
        .collect();
    assert_eq!(texte(&o), attendu);
    rc(
        &o,
        i32::from(attendu.lines().any(|l| l.starts_with("FAIL "))),
    );
    let rouges: Vec<_> = (1..=100)
        .map(|cfg| RecordTopo {
            cfg,
            flot_a2: FlotA2 {
                q_imp: vec![0.5; 4],
                ..FlotA2::default()
            },
            ..records[0].clone()
        })
        .collect();
    let texte_rouge: String = rouges
        .iter()
        .map(|r| format!("{}\n", serde_json::to_string(r).unwrap()))
        .collect();
    fs::write(&a, texte_rouge).unwrap();
    let o = d.kvu(&[
        "obs",
        "topo-gates",
        "analyse",
        "--ens",
        "q",
        "--jsonl",
        s(&a),
    ]);
    rc(&o, 1);
    assert!(texte(&o).lines().any(|l| l.starts_with("FAIL TOPO-Q ")));
    rc(
        &d.kvu(&[
            "obs",
            "topo-gates",
            "analyse",
            "--ens",
            "obs4",
            "--jsonl",
            s(&d.0.join("absent")),
        ]),
        4,
    );
}

#[test]
fn certificat_fidele_non_vert_et_erreur_de_lecture() {
    let d = Dossier::nouveau("certificat");
    let a = d.0.join("a.txt");
    let b = d.0.join("b.txt");
    let lignes = [
        "PASS TOPO-INST backend=cpu  delta=1e-15",
        "INFO OBS-4-L16-REJEU détails : écart=2e-12  ",
        "SKIP TOPO-GPU trois champs requis",
        "FAIL TOPO-Q écart > seuil",
    ];
    fs::write(
        &a,
        format!("journal ignoré\n{}\n{}\n", lignes[0], lignes[1]),
    )
    .unwrap();
    fs::write(&b, format!("{}\n{}\n", lignes[2], lignes[3])).unwrap();
    let out = d.0.join("validation/topo/topo.json");
    let liste = format!("{},{}", a.display(), b.display());
    let o = d.kvu(&[
        "obs",
        "topo-gates",
        "certificat",
        "--lignes",
        &liste,
        "--out",
        s(&out),
        "--cmd",
        "kvu obs topo-gates inst --backend cpu",
        "--cmd",
        "commande  avec espaces",
    ]);
    rc(&o, 1);
    assert_eq!(texte(&o), lignes.join("\n") + "\n");
    let v = json(&out);
    assert_eq!(v["schema"], "kvu-topo/1");
    assert_eq!(
        v["binaire_sha256"],
        sha256_fichier(Path::new(env!("CARGO_BIN_EXE_kvu"))).unwrap()
    );
    assert!(v["git_head"].is_string());
    let relues: Vec<LigneGate> = serde_json::from_value(v["lignes"].clone()).unwrap();
    assert_eq!(
        relues.iter().map(ToString::to_string).collect::<Vec<_>>(),
        lignes
    );
    assert_eq!(
        v["non_vert"],
        Value::Array(v["lignes"].as_array().unwrap()[2..].to_vec())
    );
    assert_eq!(
        v["commandes"],
        serde_json::json!([
            "kvu obs topo-gates inst --backend cpu",
            "commande  avec espaces"
        ])
    );
    let avant = fs::read(&out).unwrap();
    rc(
        &d.kvu(&[
            "obs",
            "topo-gates",
            "certificat",
            "--lignes",
            s(&d.0.join("absent")),
            "--out",
            s(&out),
        ]),
        4,
    );
    assert_eq!(fs::read(&out).unwrap(), avant);
    // SKIP seul reste rc 0, y compris dans le certificat.
    fs::write(&b, format!("{}\n", lignes[2])).unwrap();
    let o = d.kvu(&[
        "obs",
        "topo-gates",
        "certificat",
        "--lignes",
        s(&b),
        "--out",
        s(&out),
    ]);
    rc(&o, 0);
    assert_eq!(texte(&o), format!("{}\n", lignes[2]));
    assert_eq!(json(&out)["non_vert"].as_array().unwrap().len(), 1);
}

#[test]
fn gpu_moins_de_trois_skip_sans_ouvrir_le_gpu() {
    let d = Dossier::nouveau("skip");
    // Les chemins inexistants ne sont jamais lus et aucun runtime GPU n'est ouvert.
    let o = d.kvu(&[
        "obs",
        "topo-gates",
        "gpu",
        "--cfg",
        "absent1.npy,absent2.npy",
        "--backend",
        "hip",
    ]);
    rc(&o, 0);
    assert!(texte(&o).starts_with("SKIP TOPO-GPU "));
    assert_eq!(texte(&o).lines().count(), 1);
}
