//! Extensions WP9 et conservation des sorties WP8 ; aucun test historique modifié.
use kvu::commun::code;
use kvu::rg::flot::cellule::{flot, flot_options, Estimateurs, ParamFlot, Schema};
use kvu::rg::flot::etalonner::{
    depuis_json, en_json, ChaineMesuree, ParamEtalonnage, GRILLE_GELEE, GRILLE_WP9,
};
use kvu::rg::flot::extensions::{plages_bins, OptionsFlot};
use kvu::rg::flot::gates::{survie_chi2, survie_chi2_pair};
use kvu::rg::flot::jackknife::jackknife_groupes;
use kvu::rg::flot::plaq::{Courbe, PointEtalon};
use kvu::rg::flot::sd::RapportBessel;
use kvu::rg::flot::tau::StatSerie;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;

struct Dossier(PathBuf);
impl Dossier {
    fn nouveau(nom: &str) -> Self {
        let p = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("rg_wp9_{nom}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
    fn ensemble(&self) -> PathBuf {
        let ens = self.0.join("ensemble");
        for c in 1..=2 {
            let d = ens.join(format!("c{c}"));
            std::fs::create_dir_all(&d).unwrap();
            let graines = if c == 1 { [7, 11, 13] } else { [13, 7, 7] };
            for (i, graine) in graines.iter().enumerate() {
                let src = format!(
                    "{}/validation/rg/fixtures/cfg_s{graine}_q64.npy",
                    env!("CARGO_MANIFEST_DIR")
                );
                std::fs::copy(src, d.join(format!("cfg_{:06}.npy", c * 100 + i))).unwrap();
            }
            std::fs::write(
                d.join("manifest.json"),
                r#"{"beta": 2.3, "l": 9, "dump_every": 100}"#,
            )
            .unwrap();
        }
        ens
    }
    fn ecrire(&self, nom: &str, doc: &Value) -> PathBuf {
        let p = self.0.join(nom);
        std::fs::write(&p, serde_json::to_vec_pretty(doc).unwrap()).unwrap();
        p
    }
}
impl Drop for Dossier {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn param() -> ParamFlot {
    ParamFlot {
        schema: Schema::Balaban,
        pas: 1,
        estimateurs: Estimateurs {
            sd: true,
            plaq: true,
            mcrg2: true,
        },
        rapport: RapportBessel::I2SurI1,
        courbes: vec![],
        fils: 1,
    }
}

fn ajouter_courbe(par: &mut ParamFlot, refdoc: &Value) {
    let c = &refdoc["cellules"][0];
    let p = c["p_moyenne"].as_f64().unwrap();
    let r = c["r_moyenne"].as_f64().unwrap();
    par.courbes = vec![Courbe::nouvelle(
        3,
        [-0.5, -0.1, 0.2, 0.6]
            .iter()
            .enumerate()
            .map(|(i, &x)| PointEtalon {
                beta: (i + 1) as f64,
                p: p + x,
                r: r + x,
                sigma_p: 1e-4,
                sigma_r: 2e-4,
                n_chaines: 3,
            })
            .collect(),
    )
    .unwrap()];
}

/// Compare récursivement toutes les anciennes clés, y compris les flottants et les tableaux.
fn anciennes_cles(a: &Value, b: &Value) {
    match a {
        Value::Object(m) => {
            for (k, v) in m {
                anciennes_cles(v, b.get(k).unwrap_or_else(|| panic!("clé perdue : {k}")));
            }
        }
        Value::Array(v) => {
            let w = b.as_array().unwrap();
            assert_eq!(v.len(), w.len());
            for (a, b) in v.iter().zip(w) {
                anciennes_cles(a, b);
            }
        }
        _ => assert_eq!(a, b),
    }
}

fn lancer(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_kvu"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn rg_wp9_invariant_cli_api_et_cpu_explicite() {
    let d = Dossier::nouveau("invariant");
    let ens = d.ensemble();
    let par = param();
    let doc = flot(&ens, None, &par, &[]).unwrap();
    for (nom, options) in [("defaut", vec![]), ("cpu", vec!["--backend", "cpu"])] {
        let sortie = d.0.join(nom);
        let mut args = vec![
            "rg",
            "flot",
            "--ensemble",
            ens.to_str().unwrap(),
            "--sortie",
            sortie.to_str().unwrap(),
            "--fils",
            "1",
        ];
        args.extend(options);
        let o = lancer(&args);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        assert_eq!(
            std::fs::read(sortie.join("flot.json")).unwrap(),
            (serde_json::to_string_pretty(&doc).unwrap() + "\n").as_bytes()
        );
    }
    assert!(doc.get("calcul").is_none());
    assert!(doc["cellules"][0].get("configs").is_none());
    assert!(doc["cellules"][0].get("bins").is_none());
}

#[test]
fn rg_wp9_bins_detail_et_replicats_identiques_aux_chaines() {
    let d = Dossier::nouveau("bins");
    let ens = d.ensemble();
    let mut par = param();
    let sans = flot(&ens, None, &par, &[]).unwrap();
    ajouter_courbe(&mut par, &sans);
    let sans = flot(&ens, None, &par, &[]).unwrap();
    for k in [1, 2, 3, 100] {
        let options = OptionsFlot {
            bins: Some(k),
            detail_configs: true,
            ..Default::default()
        };
        let avec = flot_options(&ens, None, &par, &[], &options).unwrap();
        anciennes_cles(&sans, &avec);
        let cel = &avec["cellules"][0];
        assert_eq!(cel["bins"]["n_groupes"], if k == 1 { 6 } else { 2 });
        let cs = cel["configs"].as_array().unwrap();
        assert_eq!(cs.len(), 6);
        assert_eq!(cs[0]["chaine"], "c1");
        assert_eq!(cs[3]["chaine"], "c2");
        assert_eq!(cs[0]["fichier"], "cfg_000100.npy");
        for c in cs {
            for k in [
                "p",
                "r",
                "fraction_hors",
                "theta_max",
                "n_ambigues",
                "n_boucles",
                "sd_somme_w",
                "sd_somme_k",
            ] {
                assert!(c[k].is_number(), "{k}");
            }
        }
        for e in ["sd", "plaq", "mcrg2"] {
            let b = &cel["estimateurs"][e];
            assert_eq!(b["etat"], "ok");
            assert!(b["erreur_totale_bins"].is_number());
            assert_eq!(b["a_j"]["erreur_bins"], b["erreur_totale_bins"]);
            if k >= 3 {
                assert_eq!(
                    b["jackknife"]["replicats"],
                    b["jackknife_bins"]["replicats"]
                );
                assert_eq!(b["jackknife"]["erreur"], b["jackknife_bins"]["erreur"]);
            }
            if e == "mcrg2" {
                assert!(b["ecart_schema"]["jackknife_bins"].is_object());
                assert!(b["a_j_p"]["erreur_bins"].is_number());
                assert!(b["a_j_r"]["erreur_bins"].is_number());
            }
        }
    }
    let mut detail_seul = OptionsFlot {
        detail_configs: true,
        ..Default::default()
    };
    let detail = flot_options(&ens, None, &par, &[], &detail_seul).unwrap();
    anciennes_cles(&sans, &detail);
    assert!(detail["cellules"][0].get("bins").is_none());
    // Moyenne centrale encadrée, mais réplicats hors plage : null sans erreur.
    let p = sans["cellules"][0]["p_moyenne"].as_f64().unwrap();
    for (i, point) in par.courbes[0].points.iter_mut().enumerate() {
        point.p = p + (i as f64 - 1.5) * 1e-10;
    }
    detail_seul.bins = Some(1);
    let doc = flot_options(&ens, None, &par, &[], &detail_seul).unwrap();
    assert_eq!(doc["cellules"][0]["estimateurs"]["plaq"]["etat"], "ok");
    assert!(doc["cellules"][0]["estimateurs"]["plaq"]["jackknife_bins"].is_null());
}

#[test]
fn rg_wp9_plages_reste_et_moyenne_iid() {
    assert_eq!(plages_bins(11, 3), vec![0..3, 3..6, 6..11]);
    assert_eq!(plages_bins(2, 5), vec![0..2]);
    assert_eq!(plages_bins(6, 3), vec![0..3, 3..6]);
    assert!(plages_bins(0, 3).is_empty());
    assert!(plages_bins(3, 0).is_empty());
    let x = [0.3, 1.2, -0.7, 3.1, 2.8, -4.2];
    let groupes: Vec<&[f64]> = plages_bins(x.len(), 1).into_iter().map(|r| &x[r]).collect();
    let j = jackknife_groupes(&groupes, |g| {
        Some(
            g.iter().flat_map(|b| b.iter()).sum::<f64>()
                / g.iter().map(|b| b.len()).sum::<usize>() as f64,
        )
    })
    .unwrap();
    let moy = x.iter().sum::<f64>() / x.len() as f64;
    let variance = x.iter().map(|v| (v - moy).powi(2)).sum::<f64>() / (x.len() - 1) as f64;
    assert!((j.erreur - (variance / x.len() as f64).sqrt()).abs() < 1e-12);
}

#[test]
fn rg_wp9_protocoles_et_survie_exacte() {
    for (l, mesures, base) in [(4, 20000, 26_102_620_000), (8, 10000, 26_102_624_000)] {
        let p = ParamEtalonnage::wp9(l).unwrap();
        assert_eq!(p.betas, GRILLE_WP9);
        assert_eq!(p.betas.len(), 31);
        assert_eq!(
            (p.n_chaines, p.therm, p.mesures, p.graine_base),
            (3, 2000, mesures, base)
        );
        let j = document(&p);
        assert_eq!(j["protocole"], "wp9");
        assert_eq!(depuis_json(&j).unwrap().0, p);
    }
    assert_eq!(
        ParamEtalonnage::wp9(8).unwrap().graine(30, 9),
        26_102_627_009
    );
    assert!(ParamEtalonnage::wp9(6).is_err());
    let p = ParamEtalonnage::gele(4).unwrap();
    assert_eq!(p.betas, GRILLE_GELEE);
    assert_eq!((p.mesures, p.graine_base), (20000, 26_102_608_000));
    assert!(document(&p).get("protocole").is_none());
    for ddl in (2..=64).step_by(2) {
        for x in [0.0, 0.1, 1.0, 4.0, 14.0, 31.0, 80.0, 200.0] {
            assert!(
                (survie_chi2(x, ddl) - survie_chi2_pair(x, ddl)).abs() < 1e-12,
                "x {x} ddl {ddl}"
            );
        }
    }
    assert!((survie_chi2(1.0, 1) - 0.31731050786291415).abs() < 1e-12);
    assert!((survie_chi2(3.84145882069412, 1) - 0.05).abs() < 1e-9);
    assert!((survie_chi2(7.814727903251178, 3) - 0.05).abs() < 1e-9);
    assert_eq!(survie_chi2(f64::INFINITY, 31), 0.0);
    assert_eq!(survie_chi2(0.0, 31), 1.0);
    assert!(survie_chi2(-1.0, 31).is_nan());
    assert!(survie_chi2(1.0, 0).is_nan());
}

fn document(par: &ParamEtalonnage) -> Value {
    let mut chaines = Vec::new();
    let cs = if par.independante {
        vec![9]
    } else {
        (1..=par.n_chaines as u64).collect()
    };
    for (i, &beta) in par.betas.iter().enumerate() {
        let b = beta * par.facteur_beta;
        let p = b / (1.0 + b);
        for &c in &cs {
            let stat = |moy| StatSerie {
                moy,
                err: 1e-6,
                tau: 0.5,
                fenetre: 1,
                dtau: 0.1,
                n: par.mesures,
                sd: 1e-4,
            };
            chaines.push(ChaineMesuree {
                indice_beta: i,
                beta,
                beta_simule: b,
                graine: par.graine(i, c),
                c,
                p: stat(p),
                r: stat(p * p),
                empreinte: "synthétique".into(),
            });
        }
    }
    en_json(par, &chaines).unwrap()
}

#[test]
fn rg_wp9_etalonnage_gates_cli_et_protocoles_invalides() {
    let d = Dossier::nouveau("gates");
    let c = d.ecrire("c.json", &document(&ParamEtalonnage::wp9(4).unwrap()));
    let mut par = ParamEtalonnage::wp9(4).unwrap();
    par.independante = true;
    let mut indep = document(&par);
    let i = d.ecrire("i.json", &indep);
    par.facteur_beta = 1.05;
    let m = d.ecrire("m.json", &document(&par));
    let w = d.ecrire("w.json", &document(&ParamEtalonnage::gele(4).unwrap()));
    let o = lancer(&[
        "rg",
        "etalonnage-gates",
        "--courbe",
        c.to_str().unwrap(),
        "--independante",
        i.to_str().unwrap(),
        "--mutant",
        m.to_str().unwrap(),
        "--comparer",
        w.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{} {}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    let texte = String::from_utf8(o.stdout).unwrap();
    for id in [
        "PASS RG-PLAQ-WP9-PROTOCOLE-L4",
        "PASS RG-PLAQ-WP9-L4",
        "INFO RG-PLAQ-WP9-INTERP-L4",
        "PASS RG-PLAQ-WP9-MUT-L4",
        "INFO RG-ETAL-RECOUVREMENT-L4",
    ] {
        assert!(texte.contains(id), "{texte}");
    }
    let j: Value = serde_json::from_str(texte.lines().last().unwrap()).unwrap();
    assert_eq!(j["tout_vert"], true);
    // Mutant identique à l'indépendante : le protocole et/ou le rejet échoue.
    let o = lancer(&[
        "rg",
        "etalonnage-gates",
        "--courbe",
        c.to_str().unwrap(),
        "--independante",
        i.to_str().unwrap(),
        "--mutant",
        i.to_str().unwrap(),
    ]);
    assert_eq!(o.status.code(), Some(code::VALIDATION));
    for mutation in ["protocole", "graine", "indice", "nor", "nombre"] {
        let mut doc = indep.clone();
        match mutation {
            "protocole" => {
                doc.as_object_mut().unwrap().remove("protocole");
            }
            "graine" => doc["chaines"][0]["graine"] = json!(0),
            "indice" => doc["chaines"][0]["indice_beta"] = json!(100000),
            "nor" => doc["nor"] = json!(3),
            _ => {
                doc["chaines"].as_array_mut().unwrap().pop();
            }
        }
        d.ecrire("i.json", &doc);
        let o = lancer(&[
            "rg",
            "etalonnage-gates",
            "--courbe",
            c.to_str().unwrap(),
            "--independante",
            i.to_str().unwrap(),
        ]);
        assert_eq!(
            o.status.code(),
            Some(code::VALIDATION),
            "{mutation} : {}",
            String::from_utf8_lossy(&o.stderr)
        );
        assert!(String::from_utf8_lossy(&o.stdout).contains("FAIL RG-PLAQ-WP9-PROTOCOLE-L4"));
    }
    // Indépendante décalée mais protocole valide : gate numérique rouge.
    for ch in indep["chaines"].as_array_mut().unwrap() {
        ch["p"]["moy"] = json!(0.0);
    }
    d.ecrire("i.json", &indep);
    let o = lancer(&[
        "rg",
        "etalonnage-gates",
        "--courbe",
        c.to_str().unwrap(),
        "--independante",
        i.to_str().unwrap(),
    ]);
    assert_eq!(o.status.code(), Some(code::VALIDATION));
    assert!(String::from_utf8_lossy(&o.stdout).contains("FAIL RG-PLAQ-WP9-L4"));
}

#[test]
fn rg_wp9_refus_configuration_avant_gpu() {
    for options in [
        vec!["--backend", "hip", "--schema", "swendsen"],
        vec!["--bins", "0"],
        vec!["--backend", "inconnu"],
    ] {
        let mut args = vec!["rg", "flot", "--ensemble", "/inexistant"];
        args.extend(options);
        let o = lancer(&args);
        assert_eq!(o.status.code(), Some(code::USAGE));
        assert!(!String::from_utf8_lossy(&o.stderr).contains("dlopen"));
    }
    let o = lancer(&[
        "rg",
        "flot",
        "--ensemble",
        "/inexistant",
        "--backend",
        "hip",
        "--schema",
        "swendsen",
    ]);
    assert!(String::from_utf8_lossy(&o.stderr).contains("balaban"));
    for args in [
        vec![
            "rg",
            "etalonner",
            "--L",
            "6",
            "--protocole",
            "wp9",
            "--sortie",
            "x.json",
        ],
        vec![
            "rg",
            "etalonner",
            "--L",
            "4",
            "--protocole",
            "wp9",
            "--facteur-beta",
            "1.05",
            "--sortie",
            "x.json",
        ],
    ] {
        assert_eq!(lancer(&args).status.code(), Some(code::USAGE));
    }
}

#[test]
#[ignore = "GPU"]
fn rg_wp9_backend_auto_contre_cpu() {
    let d = Dossier::nouveau("gpu");
    let ens = d.ensemble();
    let mut par = param();
    let sans = flot(&ens, None, &par, &[]).unwrap();
    ajouter_courbe(&mut par, &sans);
    let mut doc = document(&ParamEtalonnage {
        l: 3,
        betas: par.courbes[0].points.iter().map(|p| p.beta).collect(),
        n_chaines: 1,
        therm: 0,
        mesures: 100,
        graine_base: 1,
        independante: false,
        facteur_beta: 1.0,
    });
    for (i, point) in par.courbes[0].points.iter().enumerate() {
        doc["chaines"][i]["p"]["moy"] = json!(point.p);
        doc["chaines"][i]["r"]["moy"] = json!(point.r);
    }
    let courbe = d.ecrire("courbe.json", &doc);
    let calculer = |backend: &str| {
        let sortie = d.0.join(backend);
        let o = lancer(&[
            "rg",
            "flot",
            "--ensemble",
            ens.to_str().unwrap(),
            "--backend",
            backend,
            "--courbe",
            courbe.to_str().unwrap(),
            "--bins",
            "1",
            "--detail-configs",
            "--sortie",
            sortie.to_str().unwrap(),
        ]);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        serde_json::from_slice::<Value>(&std::fs::read(sortie.join("flot.json")).unwrap()).unwrap()
    };
    let cpu = calculer("cpu");
    let gpu = calculer("auto");
    assert!(
        gpu.get("calcul").is_some(),
        "ce test exige une machine où auto choisit un GPU"
    );
    fn comparer(a: &Value, b: &Value) {
        match a {
            Value::Object(m) => {
                for (k, a) in m {
                    comparer(a, &b[k]);
                }
            }
            Value::Array(v) => {
                assert_eq!(v.len(), b.as_array().unwrap().len());
                for (a, b) in v.iter().zip(b.as_array().unwrap()) {
                    comparer(a, b);
                }
            }
            Value::Number(n) if n.is_f64() => {
                let x = n.as_f64().unwrap();
                let y = b.as_f64().unwrap();
                assert!((x - y).abs() <= 1e-9 * x.abs().max(1e-12), "{x} ≠ {y}");
            }
            _ => assert_eq!(a, b),
        }
    }
    comparer(&cpu, &gpu);
}

#[test]
fn rg_wp9_courbes_relues_l8_et_mutant_vert() {
    let d = Dossier::nouveau("courbes");
    let c4 = d.ecrire("c4.json", &document(&ParamEtalonnage::wp9(4).unwrap()));
    let c8 = d.ecrire("c8.json", &document(&ParamEtalonnage::wp9(8).unwrap()));
    let (cs, _) = kvu::rg::flot::cellule::lire_courbes(&[c4.clone(), c8.clone()]).unwrap();
    assert_eq!((cs[0].l, cs[1].l), (4, 8));
    assert_eq!(cs[0].points.len(), 31);
    let mut p = ParamEtalonnage::wp9(8).unwrap();
    p.independante = true;
    let i = document(&p);
    let fi = d.ecrire("i8.json", &i);
    let lignes = kvu::rg::flot::etalonnage_gates::batterie(&c8, &fi, None, None).unwrap();
    assert_eq!(lignes.len(), 4); // protocole, P, R (passe 2), INTERP
    assert!(lignes
        .iter()
        .all(|l| l.verdict != kvu::groupes::gates::Verdict::Fail));
    p.facteur_beta = 1.05;
    let mut m = document(&p);
    for (cm, ci) in m["chaines"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .zip(i["chaines"].as_array().unwrap())
    {
        cm["p"] = ci["p"].clone();
    }
    let fm = d.ecrire("m8.json", &m);
    let lignes = kvu::rg::flot::etalonnage_gates::batterie(&c8, &fi, Some(&fm), None).unwrap();
    let mutant = lignes
        .iter()
        .find(|l| l.id == "RG-PLAQ-WP9-MUT-L8")
        .unwrap();
    assert_eq!(mutant.verdict, kvu::groupes::gates::Verdict::Fail);
    let w = d.ecrire("w4.json", &document(&ParamEtalonnage::gele(4).unwrap()));
    assert!(kvu::rg::flot::etalonnage_gates::batterie(&c8, &fi, None, Some(&w)).is_err());
}

#[test]
fn rg_wp9_detail_premier_pas_et_schema_sans_theta() {
    use kvu::commun::ChampHote;
    use kvu::stockage::garde::GardeSsd;
    use kvu::stockage::npy::{ecrire_champ, FormatChamp};
    let d = Dossier::nouveau("deux_pas");
    let ens = d.0.join("chaine");
    std::fs::create_dir_all(&ens).unwrap();
    let mut champ = ChampHote {
        l: 27,
        q: vec![0.0; 16 * 27usize.pow(4)],
    };
    for q in champ.q.chunks_exact_mut(4) {
        q[0] = 1.0;
    }
    ecrire_champ(
        &ens.join("cfg_000001.npy"),
        &champ,
        FormatChamp::Q64,
        &GardeSsd::nouvelle(0.0),
    )
    .unwrap();
    std::fs::write(
        ens.join("manifest.json"),
        r#"{"beta":2.3,"l":27,"dump_every":100}"#,
    )
    .unwrap();
    let mut p = param();
    p.pas = 2;
    let options = OptionsFlot {
        bins: Some(10),
        detail_configs: true,
        ..Default::default()
    };
    let doc = flot_options(&ens, None, &p, &[], &options).unwrap();
    let c = &doc["cellules"][0];
    assert_eq!(c["l_fin"], 3);
    assert_eq!(c["bins"]["n_groupes"], 1);
    for cle in ["fraction_hors", "theta_max", "n_ambigues", "n_boucles"] {
        assert_eq!(c["configs"][0][cle], c["domaine"][0][cle]);
    }
    assert_ne!(c["configs"][0]["n_boucles"], c["domaine"][1]["n_boucles"]);
    let ens = d.0.join("chaine_swendsen");
    std::fs::create_dir_all(&ens).unwrap();
    ecrire_champ(
        &ens.join("cfg_000001.npy"),
        &ChampHote::froid(6).unwrap(),
        FormatChamp::Q64,
        &GardeSsd::nouvelle(0.0),
    )
    .unwrap();
    std::fs::write(
        ens.join("manifest.json"),
        r#"{"beta":2.3,"l":6,"dump_every":100}"#,
    )
    .unwrap();
    p.schema = Schema::Swendsen;
    p.pas = 1;
    let doc = flot_options(&ens, None, &p, &[], &options).unwrap();
    let c = &doc["cellules"][0];
    assert!(c.get("domaine").is_none());
    assert!(c["configs"][0].get("theta_max").is_none());
    assert!(c["configs"][0]["sd_somme_w"].is_number());
}

#[test]
fn rg_wp9_passe2_bins_plusieurs_tailles_cli_api() {
    use kvu::rg::flot::extensions::analyser_bins;
    let d = Dossier::nouveau("passe2_bins");
    let ens = d.ensemble();
    for c in 1..=2 {
        let chaine = ens.join(format!("c{c}"));
        for i in 3..8 {
            let src = chaine.join(format!("cfg_{:06}.npy", c * 100 + i % 3));
            std::fs::copy(src, chaine.join(format!("cfg_{:06}.npy", c * 100 + i))).unwrap();
        }
    }
    let mut par = param();
    let sans = flot(&ens, None, &par, &[]).unwrap();
    ajouter_courbe(&mut par, &sans);
    let calculer = |tailles: Vec<usize>| {
        flot_options(
            &ens,
            None,
            &par,
            &[],
            &OptionsFlot {
                bins_tailles: tailles,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let deux = calculer(vec![2]);
    let quatre = calculer(vec![4]);
    let plusieurs = calculer(vec![2, 4]);
    let a = &deux["cellules"][0];
    let b = &quatre["cellules"][0];
    let c = &plusieurs["cellules"][0];
    assert_eq!(
        c["bins"],
        json!({"taille": 2, "n_groupes": 8, "tailles": [
        {"taille": 2, "n_groupes": 8}, {"taille": 4, "n_groupes": 4}]})
    );
    for nom in ["sd", "plaq", "mcrg2"] {
        let aa = &a["estimateurs"][nom];
        let bb = &b["estimateurs"][nom];
        let cc = &c["estimateurs"][nom];
        assert_eq!(cc["etat"], "ok");
        for cle in [
            "valeur",
            "jackknife_bins",
            "erreur_jackknife_bins",
            "erreur_totale_bins",
            "a_j",
        ] {
            assert_eq!(aa[cle], cc[cle], "{nom}.{cle}");
        }
        assert_eq!(
            cc["jackknife_bins_tailles"][0],
            aa["jackknife_bins_tailles"][0]
        );
        assert_eq!(
            cc["jackknife_bins_tailles"][1],
            bb["jackknife_bins_tailles"][0]
        );
        assert_eq!(aa["jackknife_bins_tailles"].as_array().unwrap().len(), 1);
    }
    let ecart = |v: &Value| v["estimateurs"]["mcrg2"]["ecart_schema"].clone();
    assert_eq!(ecart(c)["jackknife_bins"], ecart(a)["jackknife_bins"]);
    assert_eq!(
        ecart(c)["jackknife_bins_tailles"][1],
        ecart(b)["jackknife_bins_tailles"][0]
    );
    for cle in ["a_j_p", "a_j_r"] {
        assert_eq!(
            c["estimateurs"]["mcrg2"][cle],
            a["estimateurs"]["mcrg2"][cle]
        );
    }
    for texte in [
        "0",
        "2,0",
        "2,2",
        "",
        "2,",
        ",2",
        "2,a",
        "-1",
        "184467440737095516160",
    ] {
        assert!(analyser_bins(texte).is_err(), "{texte}");
        let o = lancer(&["rg", "flot", "--ensemble", "/inexistant", "--bins", texte]);
        assert_eq!(o.status.code(), Some(code::USAGE), "{texte}");
    }
    assert_eq!(analyser_bins("4,2").unwrap(), vec![4, 2]);
    let sortie = d.0.join("cli");
    let o = lancer(&[
        "rg",
        "flot",
        "--ensemble",
        ens.to_str().unwrap(),
        "--bins",
        "2,4",
        "--sortie",
        sortie.to_str().unwrap(),
        "--fils",
        "1",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let cli: Value =
        serde_json::from_slice(&std::fs::read(sortie.join("flot.json")).unwrap()).unwrap();
    assert_eq!(cli["cellules"][0]["bins"], c["bins"]);
    let sd_json: Value =
        serde_json::from_str(&serde_json::to_string(&c["estimateurs"]["sd"]).unwrap()).unwrap();
    assert_eq!(cli["cellules"][0]["estimateurs"]["sd"], sd_json);
}

fn document_milieux(c: &Value) -> Value {
    use kvu::rg::flot::etalonner::courbe_depuis;
    use kvu::rg::flot::plaq::Observable;
    let (mut par, chaines) = depuis_json(c).unwrap();
    let courbe = courbe_depuis(par.l, &chaines).unwrap();
    par.activer_milieux().unwrap();
    let mut doc = document(&par);
    for (i, ch) in doc["chaines"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        ch["c"] = json!(8);
        ch["graine"] = json!(par.graine(i, 8));
        ch["p"]["moy"] = json!(
            courbe
                .evaluer(Observable::Plaquette, par.betas[i])
                .unwrap()
                .0
        );
        ch["r"]["moy"] = json!(
            courbe
                .evaluer(Observable::Rectangle, par.betas[i])
                .unwrap()
                .0
        );
    }
    let (par, chaines) = depuis_json(&doc).unwrap();
    en_json(&par, &chaines).unwrap()
}

#[test]
fn rg_wp9_passe2_param_milieux_et_refus() {
    for l in [4, 8] {
        let mut par = ParamEtalonnage::wp9(l).unwrap();
        let original = par.clone();
        assert!(!par.est_milieux());
        par.activer_milieux().unwrap();
        assert!(par.est_milieux());
        assert_eq!(par.n_chaines, 1);
        assert_eq!(par.betas.len(), 30);
        assert_eq!(par.graine(29, 8), original.graine_base + 2908);
        for (i, w) in GRILLE_WP9.windows(2).enumerate() {
            assert_eq!(par.betas[i], (w[0] + w[1]) / 2.0);
        }
        let c = document(&original);
        assert!(c.get("milieux").is_none());
        let m = document_milieux(&c);
        assert_eq!(m["milieux"], true);
        assert_eq!(m["n_chaines"], 1);
        assert_eq!(depuis_json(&m).unwrap().0, par);
        let d = Dossier::nouveau(&format!("milieux_l{l}"));
        let f = d.ecrire("m.json", &m);
        assert!(kvu::rg::flot::cellule::lire_courbes(&[f]).is_err());
        par.independante = true;
        assert!(!par.est_milieux());
        par.independante = false;
        par.facteur_beta = 1.05;
        assert!(!par.est_milieux());
    }
    let mut par = ParamEtalonnage::gele(4).unwrap();
    assert!(par.activer_milieux().is_err());
    for options in [
        vec!["--independante"],
        vec!["--facteur-beta", "1.0"],
        vec!["--facteur-beta", "1.05"],
        vec!["--protocole", "wp8"],
    ] {
        let mut args = vec![
            "rg",
            "etalonner",
            "--L",
            "4",
            "--milieux",
            "--sortie",
            "/inexistant/m.json",
        ];
        args.extend(options);
        assert_eq!(lancer(&args).status.code(), Some(code::USAGE));
    }
    // Calcul réel court : la chaîne c = 8 et sa graine sont bien choisies par le moteur.
    let mut par = ParamEtalonnage::wp9(4).unwrap();
    par.activer_milieux().unwrap();
    par.therm = 2;
    par.mesures = 100;
    let cs = kvu::rg::flot::etalonner::etalonner(&par, 2).unwrap();
    assert_eq!(cs.len(), 30);
    for (i, ch) in cs.iter().enumerate() {
        assert_eq!(ch.c, 8);
        assert_eq!(ch.graine, par.graine(i, 8));
        assert_eq!(ch.beta_simule, par.betas[i]);
    }
}

#[test]
fn rg_wp9_passe2_gates_rectangle_milieux_et_protocoles() {
    use kvu::groupes::gates::Verdict;
    use kvu::rg::flot::etalonnage_gates::batterie_milieux;
    let d = Dossier::nouveau("passe2_gates");
    for l in [4, 8] {
        let mut par = ParamEtalonnage::wp9(l).unwrap();
        let courbe = document(&par);
        let c = d.ecrire("c.json", &courbe);
        par.independante = true;
        let independante = document(&par);
        let i = d.ecrire("i.json", &independante);
        let milieux = document_milieux(&courbe);
        let m = d.ecrire("m.json", &milieux);
        let lignes = batterie_milieux(&c, &i, None, None, Some(&m)).unwrap();
        for id in [
            format!("RG-PLAQ-WP9-R-L{l}"),
            format!("RG-PLAQ-WP9-MILIEU-P-L{l}"),
            format!("RG-PLAQ-WP9-MILIEU-R-L{l}"),
        ] {
            let ligne = lignes.iter().find(|x| x.id == id).unwrap();
            assert_eq!(ligne.verdict, Verdict::Pass, "{}", ligne.detail);
            assert!(ligne.detail.contains("3.59"));
            assert!(ligne.detail.contains("β="));
            if id.contains("MILIEU") {
                assert!(ligne.detail.contains("ddl = 30"));
            }
        }
        let o = lancer(&[
            "rg",
            "etalonnage-gates",
            "--courbe",
            c.to_str().unwrap(),
            "--independante",
            i.to_str().unwrap(),
            "--milieux",
            m.to_str().unwrap(),
            "--json",
        ]);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        for obs in ["p", "r"] {
            let mut perturbe = milieux.clone();
            let sigma = 1e-6_f64.hypot(1e-6 / 3.0_f64.sqrt());
            let ch = &mut perturbe["chaines"][10][obs]["moy"];
            *ch = json!(ch.as_f64().unwrap() + 10.0 * sigma);
            d.ecrire("m.json", &perturbe);
            let lignes = batterie_milieux(&c, &i, None, None, Some(&m)).unwrap();
            let id = format!("RG-PLAQ-WP9-MILIEU-{}-L{l}", obs.to_uppercase());
            assert_eq!(
                lignes.iter().find(|x| x.id == id).unwrap().verdict,
                Verdict::Fail
            );
        }
        for mutation in [
            "flag",
            "independante",
            "l",
            "base",
            "therm",
            "mesures",
            "nor",
            "facteur",
            "beta",
            "beta_chaine",
            "graine",
            "c",
            "nombre",
            "indice",
            "n",
        ] {
            let mut faux = milieux.clone();
            match mutation {
                "flag" => {
                    faux.as_object_mut().unwrap().remove("milieux");
                }
                "independante" => faux["independante"] = json!(true),
                "l" => faux["l"] = json!(6),
                "base" => faux["graine_base"] = json!(1),
                "therm" => faux["therm"] = json!(1),
                "mesures" => faux["mesures"] = json!(100),
                "nor" => faux["nor"] = json!(3),
                "facteur" => faux["facteur_beta"] = json!(1.05),
                "beta" => faux["betas"][0] = json!(0.4),
                "beta_chaine" => faux["chaines"][0]["beta"] = json!(0.4),
                "graine" => faux["chaines"][0]["graine"] = json!(1),
                "c" => faux["chaines"][0]["c"] = json!(9),
                "nombre" => {
                    faux["chaines"].as_array_mut().unwrap().pop();
                }
                "indice" => faux["chaines"][0]["indice_beta"] = json!(1),
                _ => faux["chaines"][0]["p"]["n"] = json!(1),
            }
            d.ecrire("m.json", &faux);
            let lignes = batterie_milieux(&c, &i, None, None, Some(&m)).unwrap();
            for ligne in lignes.iter().filter(|x| x.id.contains("MILIEU")) {
                assert_eq!(ligne.verdict, Verdict::Fail, "{mutation}");
                assert!(ligne.detail.contains("protocole milieux invalide"));
            }
        }
        // R seul perturbé : la nouvelle gate utilise bien la mesure du rectangle.
        let mut faux = independante.clone();
        faux["chaines"][10]["r"]["moy"] = json!(0.0);
        d.ecrire("i.json", &faux);
        let lignes = batterie_milieux(&c, &i, None, None, None).unwrap();
        assert_eq!(
            lignes
                .iter()
                .find(|x| x.id == format!("RG-PLAQ-WP9-R-L{l}"))
                .unwrap()
                .verdict,
            Verdict::Fail
        );
        assert_eq!(
            lignes
                .iter()
                .find(|x| x.id == format!("RG-PLAQ-WP9-L{l}"))
                .unwrap()
                .verdict,
            Verdict::Pass
        );
    }
}

#[test]
fn rg_wp9_passe2_seuil_bonferroni_et_sigma_milieux_maximum() {
    use kvu::groupes::gates::Verdict;
    use kvu::rg::flot::etalonnage_gates::{batterie_milieux, SEUIL_Z_WP9};
    let d = Dossier::nouveau("passe2_seuil");
    assert_eq!(SEUIL_Z_WP9, 3.59);
    let mut par = ParamEtalonnage::wp9(4).unwrap();
    let mut courbe = document(&par);
    // Les erreurs des deux nœuds voisins sont différentes ; le maximum est requis.
    for ch in courbe["chaines"].as_array_mut().unwrap() {
        if ch["indice_beta"] == 11 {
            ch["p"]["err"] = json!(3e-6);
        }
    }
    let c = d.ecrire("c.json", &courbe);
    par.independante = true;
    let mut indep = document(&par);
    let sigma = 1e-6_f64.hypot(1e-6 / 3.0_f64.sqrt());
    let valeur = indep["chaines"][10]["p"]["moy"].as_f64().unwrap();
    indep["chaines"][10]["p"]["moy"] = json!(valeur + 3.3 * sigma);
    let i = d.ecrire("i.json", &indep);
    let mut milieux = document_milieux(&courbe);
    let sigma_max = 1e-6_f64.hypot(3e-6 / 3.0_f64.sqrt());
    let valeur = milieux["chaines"][10]["p"]["moy"].as_f64().unwrap();
    milieux["chaines"][10]["p"]["moy"] = json!(valeur + 3.3 * sigma_max);
    let m = d.ecrire("m.json", &milieux);
    let lignes = batterie_milieux(&c, &i, None, None, Some(&m)).unwrap();
    for id in ["RG-PLAQ-WP9-L4", "RG-PLAQ-WP9-MILIEU-P-L4"] {
        let ligne = lignes.iter().find(|x| x.id == id).unwrap();
        assert_eq!(ligne.verdict, Verdict::Pass, "{}", ligne.detail);
    }
    let milieu = lignes
        .iter()
        .find(|x| x.id == "RG-PLAQ-WP9-MILIEU-P-L4")
        .unwrap();
    assert!(
        milieu.detail.contains("max|z| = 3.300000"),
        "{}",
        milieu.detail
    );
    // Au-dessus de 3,59 le milieu est rouge, même avec un χ² global acceptable.
    milieux["chaines"][10]["p"]["moy"] = json!(valeur + 3.7 * sigma_max);
    d.ecrire("m.json", &milieux);
    let lignes = batterie_milieux(&c, &i, None, None, Some(&m)).unwrap();
    assert_eq!(
        lignes
            .iter()
            .find(|x| x.id == "RG-PLAQ-WP9-MILIEU-P-L4")
            .unwrap()
            .verdict,
        Verdict::Fail
    );
}
