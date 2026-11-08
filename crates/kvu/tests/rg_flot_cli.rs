//! `rg flot` (WP8) sur un petit ensemble synthétique (3 chaînes de 3 fixtures 9⁴ → 3⁴) : structure du flot.json, hors étalonnage sans courbe,
//! inversion avec courbe synthétique, jackknife par chaîne, indépendance au nombre de fils, refus d'écraser, usages invalides.

use kvu::rg::flot::cellule::{flot, lire_courbes, Estimateurs, ParamFlot, Schema};
use kvu::rg::flot::etalonner::{en_json, ChaineMesuree, ParamEtalonnage};
use kvu::rg::flot::sd::RapportBessel;
use kvu::rg::flot::tau::stat_serie;
use std::path::{Path, PathBuf};
use std::process::Command;

fn dossier(nom: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("rg_flot_{nom}_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Ensemble synthétique : ens/c{1,2,3}/cfg_000100.npy.. (fixtures 9⁴ de graines 7, 11, 13, chacune en trois fichiers) + manifest.json.
fn ensemble(nom: &str) -> PathBuf {
    let ens = dossier(nom);
    for (i, graine) in [7, 11, 13].iter().enumerate() {
        let c = ens.join(format!("c{}", i + 1));
        std::fs::create_dir_all(&c).unwrap();
        let src = format!("{}/validation/rg/fixtures/cfg_s{graine}_q64.npy", env!("CARGO_MANIFEST_DIR"));
        for k in 1..=3 {
            std::fs::copy(&src, c.join(format!("cfg_{:06}.npy", 100 * k))).unwrap();
        }
        std::fs::write(c.join("manifest.json"), r#"{"beta": 2.3, "l": 9, "dump_every": 100}"#).unwrap();
    }
    ens
}

fn serie_stat(v: f64) -> kvu::rg::flot::tau::StatSerie {
    let mut s = stat_serie(&(0..64).map(|i| v + 1e-4 * ((i * 7 % 13) as f64 - 6.0)).collect::<Vec<_>>());
    s.err = 1e-4;
    s
}

/// Moyennes (P̄, R̄) de la cellule synthétique après un pas de Bałaban (lues sur un premier flot sans courbe).
fn moyennes(ens: &Path) -> (f64, f64) {
    let doc = flot(ens, None, &param(vec![], 1), &[]).unwrap();
    let c = &doc["cellules"][0];
    (c["p_moyenne"].as_f64().unwrap(), c["r_moyenne"].as_f64().unwrap())
}

/// Courbe d'étalonnage synthétique de côté 3 : P et R croissantes avec β.
fn fichier_courbe(nom: &str, betas: &[f64], p: &[f64], r: &[f64]) -> PathBuf {
    let par = ParamEtalonnage { l: 3, betas: betas.to_vec(), n_chaines: 1, therm: 0, mesures: 64, graine_base: 1, independante: false, facteur_beta: 1.0 };
    let chaines: Vec<ChaineMesuree> = (0..betas.len())
        .map(|i| ChaineMesuree {
            indice_beta: i,
            beta: betas[i],
            beta_simule: betas[i],
            graine: i as u64,
            c: 1,
            p: serie_stat(p[i]),
            r: serie_stat(r[i]),
            empreinte: String::new(),
        })
        .collect();
    let f = dossier(nom).join("courbe.json");
    std::fs::write(&f, serde_json::to_string(&en_json(&par, &chaines).unwrap()).unwrap()).unwrap();
    f
}

fn param(courbes: Vec<kvu::rg::flot::plaq::Courbe>, fils: usize) -> ParamFlot {
    ParamFlot { schema: Schema::Balaban, pas: 1, estimateurs: Estimateurs { sd: true, plaq: true, mcrg2: true }, rapport: RapportBessel::I2SurI1, courbes, fils }
}

#[test]
fn sans_courbe_sd_seul_et_hors_etalonnage() {
    let ens = ensemble("sanscourbe");
    let doc = flot(&ens, None, &param(vec![], 2), &[]).unwrap();
    assert_eq!(doc["format"], "kvu-rg-flot/1");
    let c = &doc["cellules"][0];
    assert_eq!(c["n_chaines"], 3);
    assert_eq!(c["n_configs"], 9);
    assert_eq!(c["l_entree"], 9);
    assert_eq!(c["l_fin"], 3);
    assert_eq!(c["estimateurs"]["sd"]["etat"], "ok");
    assert_eq!(c["estimateurs"]["plaq"]["etat"], "hors_etalonnage");
    assert_eq!(c["estimateurs"]["mcrg2"]["etat"], "hors_etalonnage");
    assert!(c["domaine"][0]["fraction_hors"].is_number());
    // trois fixtures différentes : l'erreur jackknife par chaîne est finie et strictement positive
    let e = c["estimateurs"]["sd"]["erreur_jackknife"].as_f64().unwrap();
    assert!(e.is_finite() && e > 0.0, "{e}");
}

#[test]
fn avec_courbe_inversion_jackknife_et_a_j() {
    let ens = ensemble("aveccourbe");
    let (pb, rb) = moyennes(&ens);
    let f = fichier_courbe("c1", &[1.0, 2.0, 4.0, 8.0], &[pb - 0.3, pb - 0.1, pb + 0.2, pb + 0.5], &[rb - 0.2, rb - 0.1, rb + 0.3, rb + 0.6]);
    let (courbes, fichiers) = lire_courbes(&[f]).unwrap();
    let doc = flot(&ens, None, &param(courbes, 1), &fichiers).unwrap();
    let c = &doc["cellules"][0];
    let p = &c["estimateurs"]["plaq"];
    assert_eq!(p["etat"], "ok", "{p}");
    let b = p["valeur"].as_f64().unwrap();
    assert!(b > 1.0 && b < 8.0);
    assert!(p["a_j"]["a_j"].as_f64().is_some());
    assert_eq!(c["estimateurs"]["mcrg2"]["etat"], "ok");
    assert!(c["estimateurs"]["mcrg2"]["ecart_schema"]["valeur"].as_f64().unwrap() >= 0.0);
    assert_eq!(p["jackknife"]["replicats"].as_array().unwrap().len(), 3);
    assert_eq!(doc["courbes"][0]["l"], 3);
}

#[test]
fn hors_plage_aucune_extrapolation() {
    let ens = ensemble("horsplage");
    // P et R de la courbe bien en dessous des moyennes de la cellule : cible au-dessus de la plage
    let (pb, rb) = moyennes(&ens);
    let f = fichier_courbe("c2", &[1.0, 2.0, 4.0], &[pb - 0.5, pb - 0.4, pb - 0.3], &[rb - 0.5, rb - 0.4, rb - 0.3]);
    let (courbes, fichiers) = lire_courbes(&[f]).unwrap();
    let doc = flot(&ens, None, &param(courbes, 1), &fichiers).unwrap();
    let e = &doc["cellules"][0]["estimateurs"];
    assert_eq!(e["plaq"]["etat"], "hors_etalonnage");
    assert_eq!(e["plaq"]["cote"], "haut");
    assert!(e["plaq"].get("valeur").is_none());
    assert_eq!(e["mcrg2"]["etat"], "hors_etalonnage");
}

#[test]
fn sortie_independante_du_nombre_de_fils() {
    let ens = ensemble("fils");
    let (pb, rb) = moyennes(&ens);
    let f = fichier_courbe("c3", &[1.0, 2.0, 4.0, 8.0], &[pb - 0.3, pb - 0.1, pb + 0.2, pb + 0.5], &[rb - 0.2, rb - 0.1, rb + 0.3, rb + 0.6]);
    let (courbes, fichiers) = lire_courbes(&[f]).unwrap();
    let a = serde_json::to_string_pretty(&flot(&ens, None, &param(courbes.clone(), 1), &fichiers).unwrap()).unwrap();
    let b = serde_json::to_string_pretty(&flot(&ens, None, &param(courbes, 7), &fichiers).unwrap()).unwrap();
    assert_eq!(a, b);
}

#[test]
fn cli_ecrit_flot_json_et_refuse_d_ecraser() {
    let ens = ensemble("cli");
    let sortie = dossier("cli_sortie");
    let _ = std::fs::remove_file(sortie.join("flot.json"));
    let lancer = || {
        Command::new(env!("CARGO_BIN_EXE_kvu"))
            .args(["rg", "flot", "--ensemble", ens.to_str().unwrap(), "--estimateurs", "sd", "--sortie", sortie.to_str().unwrap(), "--fils", "2"])
            .output()
            .unwrap()
    };
    let o = lancer();
    assert_eq!(o.status.code(), Some(0), "{}", String::from_utf8_lossy(&o.stderr));
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(sortie.join("flot.json")).unwrap()).unwrap();
    assert_eq!(v["cellules"].as_array().unwrap().len(), 1);
    let o2 = lancer();
    assert_ne!(o2.status.code(), Some(0));
    // usages invalides
    let o3 = Command::new(env!("CARGO_BIN_EXE_kvu")).args(["rg", "flot", "--ensemble", ens.to_str().unwrap(), "--estimateurs", "zz"]).output().unwrap();
    assert_ne!(o3.status.code(), Some(0));
    let o4 = Command::new(env!("CARGO_BIN_EXE_kvu")).args(["rg", "etalonner", "--L", "5", "--sortie", "/nonexistent/x.json"]).output().unwrap();
    assert_ne!(o4.status.code(), Some(0));
    let o5 = Command::new(env!("CARGO_BIN_EXE_kvu")).args(["rg", "etalonner", "--L", "4", "--facteur-beta", "1.05", "--sortie", "/nonexistent/x.json"]).output().unwrap();
    assert_ne!(o5.status.code(), Some(0), "le mutant sans --independante doit être refusé");
}
