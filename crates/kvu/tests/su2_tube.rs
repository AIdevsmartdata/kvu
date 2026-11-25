//! Module tube (EXPÉRIMENTAL) : `su2 gen --cotes`, `su2 tube smin | apparie`. Tests de la révision REV2 du 11/11/2026
//! (relecture `revues/tube-claude/revue.md`, objections citées par leur numéro). Réseaux minuscules : la suite tient en
//! quelques secondes en `--release`.

use kvu::commun::{Erreur, Resultat};
use kvu::heritage::manifest::sha256_fichier;
use kvu::stockage::garde::GardeSsd;
use kvu::stockage::provenance::Manifeste;
use kvu::su2::torsion::Torsion;
use kvu::su2::tube::cli::{graine_tache, sous_manifeste};
use kvu::su2::tube::obs::moyenne_erreur;
use kvu::su2::tube::reseau::{ChaineTube, GeoA, Pile};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Code de retour d'une erreur d'usage (`Erreur::Config`).
const RC_USAGE: i32 = 2;
/// Ligne d'avertissement écrite sur stderr au lancement (étiquette EXPÉRIMENTALE, mêmes mots que le module intrication).
const AVERTISSEMENT: &str =
    "module expérimental : interface et formats susceptibles de changer ; résultats à confirmer indépendamment";
/// S/β minimale du secteur doublement tordu (ℓ, L) = (4, 2) : référence G1 du RAPPORT du module (9 décimales).
const SMIN_4_2: f64 = 6.740116866;

fn dossier_test(nom: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("kvu_su2_tube_{nom}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Code rendu par [`kvu_lancer`] quand le délai est dépassé (l'enfant, le nôtre, est alors tué).
const RC_DELAI: i32 = -2;
/// Code rendu quand l'enfant est terminé par un signal.
const RC_SIGNAL: i32 = -1;

/// Lance `kvu <args>` dans un sous-processus isolé (variables KEVINOTRON_* / KVU_* héritées retirées, seuils RAM et SSD de
/// test nuls sauf `env`, 2 fils), répertoire courant `cwd`, au plus `delai_s` secondes : au-delà, l'enfant est tué et le code
/// rendu est [`RC_DELAI`] (un refus attendu qui ne vient pas ne bloque pas la suite). Rend (code, stdout, stderr).
fn kvu_lancer(
    cwd: &Path,
    args: &[&str],
    env: &[(&str, &str)],
    delai_s: u64,
) -> (i32, String, String) {
    use std::io::Read;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};
    let mut c = Command::new(env!("CARGO_BIN_EXE_kvu"));
    for (k, _) in
        std::env::vars().filter(|(k, _)| k.starts_with("KEVINOTRON_") || k.starts_with("KVU_"))
    {
        c.env_remove(k);
    }
    c.env("KEVINOTRON_VERROU", "ignorer")
        .env("KEVINOTRON_RAM_MIN_GO", "0")
        .env("KEVINOTRON_SSD_MIN_GO", "0")
        .env("RAYON_NUM_THREADS", "2");
    for (k, v) in env {
        c.env(k, v);
    }
    let mut enfant = c
        .current_dir(cwd)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let lire = |mut f: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut b = Vec::new();
            let _ = f.read_to_end(&mut b);
            String::from_utf8_lossy(&b).into_owned()
        })
    };
    let t_out = lire(Box::new(enfant.stdout.take().unwrap()));
    let t_err = lire(Box::new(enfant.stderr.take().unwrap()));
    let debut = Instant::now();
    let rc = loop {
        if let Some(st) = enfant.try_wait().unwrap() {
            break st.code().unwrap_or(RC_SIGNAL);
        }
        if debut.elapsed() > Duration::from_secs(delai_s) {
            let _ = enfant.kill();
            let _ = enfant.wait();
            break RC_DELAI;
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    (rc, t_out.join().unwrap(), t_err.join().unwrap())
}

/// [`kvu_lancer`] sans variable de plus, délai de 900 s (runs normaux du module : moins d'une seconde sur un disque libre ;
/// les fsync du manifeste peuvent attendre des dizaines de secondes sur un disque saturé).
fn kvu_dans(cwd: &Path, args: &[&str]) -> (i32, String, String) {
    kvu_lancer(cwd, args, &[], 900)
}

fn lire_json(p: &Path) -> Value {
    serde_json::from_str(
        &std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{} : {e}", p.display())),
    )
    .unwrap()
}

fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// Manifeste clos avec succès, COMPLETE écrit EN DERNIER (il porte le sha256 du manifeste final), fichiers consignés intacts.
fn verifier_run_complet(run: &Path) -> Value {
    let m = lire_json(&run.join("manifest.json"));
    assert_eq!(m["statut"], "termine", "{m}");
    assert_eq!(m["rc"], 0);
    let c = std::fs::read_to_string(run.join("COMPLETE")).expect("COMPLETE absent");
    let sha = sha256_fichier(&run.join("manifest.json")).unwrap();
    assert!(
        c.contains(&format!("manifest_sha256={sha}")),
        "COMPLETE n'est pas écrit après le manifeste final : {c}"
    );
    for f in m["fichiers"].as_array().expect("fichiers") {
        let p = run.join(f["nom"].as_str().unwrap());
        assert_eq!(
            sha256_fichier(&p).unwrap(),
            f["sha256"].as_str().unwrap(),
            "{}",
            p.display()
        );
    }
    m
}

/// Refus d'usage : code 2, message de kvu (pas une erreur d'analyse de clap), rien d'écrit sous `run`.
fn verifier_refus(d: &Path, run: &Path, args: &[&str], fragment: &str) {
    verifier_refus_rc(d, run, args, &[], RC_USAGE, fragment, 30);
}

/// Refus de code `rc_attendu`, sous l'environnement `env`, dans le délai `delai_s` (un ancien chemin qui boucle est tué).
fn verifier_refus_rc(
    d: &Path,
    run: &Path,
    args: &[&str],
    env: &[(&str, &str)],
    rc_attendu: i32,
    fragment: &str,
    delai_s: u64,
) {
    let (rc, _, err) = kvu_lancer(d, args, env, delai_s);
    assert_eq!(rc, rc_attendu, "{args:?} : {err}");
    assert!(!err.contains("panicked"), "{args:?} : panique : {err}");
    assert!(
        !err.contains("error:"),
        "{args:?} : refus de clap, pas de kvu : {err}"
    );
    assert!(
        err.contains(fragment),
        "{args:?} : message sans « {fragment} » : {err}"
    );
    assert!(!run.exists(), "{args:?} : {} a été créé", run.display());
}

// ------------------------------------------------------------------------------------------
// su2 gen --cotes : manifeste, géométrie, statut, COMPLETE en dernier
// ------------------------------------------------------------------------------------------

#[test]
fn gen_cotes_manifeste_geometrie_statut_complete() {
    let d = dossier_test("gen");
    let run = d.join("run");
    let (rc, out, err) = kvu_dans(
        &d,
        &[
            "su2",
            "gen",
            "--cotes",
            "4,4,2,2",
            "--backend",
            "cpu",
            "--torsion",
            "01,23",
            "--start",
            "vide",
            "--beta",
            "2.0",
            "--graine",
            "3",
            "--therm",
            "4",
            "--nmeas",
            "20",
            "--nor",
            "1",
            "--dump-every",
            "10",
            "--mesures",
            "tube:01:2:1",
            "--outdir",
            &s(&run),
        ],
    );
    assert_eq!(rc, 0, "{out}\n{err}");
    assert!(err.contains(AVERTISSEMENT), "{err}");
    let m = verifier_run_complet(&run);
    assert_eq!(m["mode"], "su2-gen-cotes");
    assert_eq!(m["parametres"]["statut"], "experimental", "{m}");
    let g = &m["geometrie"];
    assert_eq!(g["schema"], "kvu-geometrie/1", "{g}");
    assert_eq!(g["cotes"], json!([4, 4, 2, 2]));
    assert_eq!(
        lire_json(&run.join("geometrie.json")),
        *g,
        "geometrie.json ≠ bloc du manifeste"
    );
    let noms: HashSet<&str> = m["fichiers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["nom"].as_str().unwrap())
        .collect();
    for n in [
        "therm.csv",
        "measurements.csv",
        "tube.csv",
        "resume.json",
        "geometrie.json",
        "torsion.json",
        "cfg_000010.npy",
        "cfg_000020.npy",
    ] {
        assert!(noms.contains(n), "{n} non consigné : {noms:?}");
    }
    let r = lire_json(&run.join("resume.json"));
    assert_eq!(r["chaines"][0]["n_obs"], 20);
    assert!(
        r["chaines"][0]["V"][1]
            .as_f64()
            .is_some_and(|e| e.is_finite() && e >= 0.0),
        "{r}"
    );
}

#[test]
fn erreur_apres_ouverture_du_manifeste_le_clot_en_echec() {
    // objection 10 : mécanisme commun à gen --cotes et apparie (aucun déclencheur par la ligne de commande sans injection de
    // faute : la faute est injectée ici dans le corps)
    let d = dossier_test("sous_manifeste");
    let garde = GardeSsd::nouvelle(0.0);
    let run = d.join("echec");
    let man =
        Manifeste::ouvrir_avec_garde(&run, "essai", &["essai".to_string()], &garde, 0).unwrap();
    let r: Resultat<()> = sous_manifeste(man, |man| {
        man.champ("mode", "\"essai\"")?;
        std::fs::write(run.join("x.txt"), b"x").unwrap();
        Err(Erreur::Validation(
            "faute injectée après les chaînes".into(),
        ))
    });
    let e = r.unwrap_err();
    let m = lire_json(&run.join("manifest.json"));
    assert_eq!(m["statut"], "echec", "{m}");
    assert_eq!(m["rc"], json!(e.code_sortie()));
    assert!(
        m["raison_echec"]
            .as_str()
            .unwrap_or("")
            .contains("faute injectée"),
        "{m}"
    );
    assert!(!run.join("COMPLETE").exists());
    let run = d.join("succes");
    let man =
        Manifeste::ouvrir_avec_garde(&run, "essai", &["essai".to_string()], &garde, 0).unwrap();
    sous_manifeste(man, |_| Ok(())).unwrap();
    verifier_run_complet(&run);
}

// ------------------------------------------------------------------------------------------
// Coloriage : balayage en place, site par site, contre sweep_hb (deux phases par classe)
// ------------------------------------------------------------------------------------------

/// Contrôle structurel du coloriage, repris de `geom::reseau::tests::coloriage_valide` (socle) : les classes partitionnent les
/// sites et deux voisins (μ = 0..3, voisin du haut ; le voisin du bas est le voisin du haut d'un autre site) ne partagent pas
/// de classe.
fn classes_independantes(g: &GeoA, couleurs: &[Vec<usize>]) -> Result<(), String> {
    let mut col = vec![usize::MAX; g.v];
    for (k, cl) in couleurs.iter().enumerate() {
        for &x in cl {
            if col[x] != usize::MAX {
                return Err(format!("site {x} dans deux classes"));
            }
            col[x] = k;
        }
    }
    if let Some(x) = col.iter().position(|&c| c == usize::MAX) {
        return Err(format!("site {x} sans classe"));
    }
    for x in 0..g.v {
        for (mu, &xm) in g.coords(x).iter().enumerate() {
            let y = g.nb_up(x, mu, xm);
            if col[x] == col[y] {
                return Err(format!(
                    "voisins {x} et {y} (μ = {mu}) dans la classe {}",
                    col[x]
                ));
            }
        }
    }
    Ok(())
}

#[test]
fn coloriage_balayage_en_place_egal_a_sweep_hb() {
    for (cotes, torsion) in [
        ([3, 2, 5, 2], "01,23"),
        ([6, 4, 3, 2], "01,23"),
        ([6, 4, 3, 2], "23"),
    ] {
        let g = GeoA::nouvelle(cotes).unwrap();
        let pile = Pile::depuis_torsion(&g, &Torsion::analyser(torsion).unwrap());
        let mut a = ChaineTube::nouvelle(g.clone(), 2026_1111);
        a.init(true);
        // les classes partitionnent les sites
        let mut vu = vec![0u32; g.v];
        for cl in &a.couleurs {
            for &x in cl {
                vu[x] += 1;
            }
        }
        assert!(
            vu.iter().all(|&n| n == 1),
            "{cotes:?} : classes qui ne partitionnent pas les sites"
        );
        // contrôle STRUCTUREL (celui du socle, geom::reseau::tests::coloriage_valide) : aucune classe ne contient deux voisins
        classes_independantes(&g, &a.couleurs).unwrap_or_else(|e| panic!("{cotes:?} : {e}"));
        let mut b = a.clone();
        for _ in 0..3 {
            a.sweep_hb(2.3, &pile);
            b.sweep_hb_en_place(2.3, &pile);
        }
        assert!(
            a.u == b.u,
            "{cotes:?} {torsion} : balayage par classes ≠ balayage en place"
        );
        assert_eq!(a.n_tries, b.n_tries);
        // MUTANT (rouge attendu) : une seule classe, donc des voisins dans la même classe ⇒ refusé par le contrôle structurel,
        // et ici les deux balayages diffèrent aussi (l'égalité des balayages seule ne suffit pas en général : un champ froid sans
        // torsion à β très grand peut rester identique dans les deux méthodes ; le contrôle structurel, lui, ne dépend pas du champ)
        let mut m1 = ChaineTube::nouvelle(g.clone(), 2026_1111);
        m1.init(true);
        m1.couleurs = vec![(0..g.v).collect()];
        assert!(
            classes_independantes(&g, &m1.couleurs).is_err(),
            "{cotes:?} : le contrôle structurel accepte une classe unique"
        );
        let mut m2 = m1.clone();
        m1.sweep_hb(2.3, &pile);
        m2.sweep_hb_en_place(2.3, &pile);
        assert!(
            m1.u != m2.u,
            "{cotes:?} : le mutant « une seule classe » n'est pas détecté"
        );
    }
}

// ------------------------------------------------------------------------------------------
// Erreurs de blocs (objections 1 et 3)
// ------------------------------------------------------------------------------------------

#[test]
fn erreur_de_blocs_non_degeneree() {
    // 4 blocs de 2 : moyennes 2, 2, 6, 0 ; moyenne 5/2 ; Σ(m_b − 5/2)² = 1/4 + 1/4 + 49/4 + 25/4 = 19 ;
    // erreur = √(19 / (4·3)) [calcul à la main]. Contre-exemples de la relecture : /nb ⇒ √(19/4), /(nb·nb) ⇒ √(19/16).
    let x = [1.0, 3.0, 2.0, 2.0, 5.0, 7.0, 0.0, 0.0];
    let (m, e) = moyenne_erreur(&x, 4);
    assert_eq!(m, 2.5);
    assert!((e - (19.0f64 / 12.0).sqrt()).abs() < 1e-15, "erreur {e}");
}

#[test]
fn moyenne_et_erreur_sur_les_memes_mesures() {
    // objection 3 : 9 = 4·2 + 1 ; la 9ᵉ valeur n'entre ni dans les blocs ni dans la moyenne
    let x = [1.0, 3.0, 2.0, 2.0, 5.0, 7.0, 0.0, 0.0, 100.0];
    let (m, e) = moyenne_erreur(&x, 4);
    assert_eq!(m, 2.5, "la moyenne inclut des mesures hors des blocs");
    assert!((e - (19.0f64 / 12.0).sqrt()).abs() < 1e-15, "erreur {e}");
}

// ------------------------------------------------------------------------------------------
// smin : référence G1 et mutants (rouges attendus sur S_min)
// ------------------------------------------------------------------------------------------

fn smin(d: &Path, args: &[&str]) -> (Value, String) {
    let mut a = vec!["su2", "tube", "smin"];
    a.extend_from_slice(args);
    let (rc, out, err) = kvu_dans(d, &a);
    assert_eq!(rc, 0, "{a:?} : {err}");
    (
        serde_json::from_str(out.lines().last().unwrap()).unwrap(),
        err,
    )
}

#[test]
fn smin_reference_et_mutants_rouges() {
    let d = dossier_test("smin");
    let (j, err) = smin(&d, &["--cotes", "4,4,2,2"]);
    assert!(err.contains(AVERTISSEMENT), "{err}");
    let s0 = j["S_sur_beta"].as_f64().unwrap();
    assert!((s0 - SMIN_4_2).abs() < 1e-8, "S_min(4,2) = {s0}");
    assert!(j["grad_max"].as_f64().unwrap() < 1e-10);
    for mutant in ["pile-incomplete", "mauvais-plan"] {
        let (jm, _) = smin(&d, &["--cotes", "4,4,2,2", "--mutant", mutant]);
        let sm = jm["S_sur_beta"].as_f64().unwrap();
        assert!(
            (sm - SMIN_4_2).abs() > 0.5,
            "mutant {mutant} non détecté : S/β = {sm}"
        );
    }
}

// ------------------------------------------------------------------------------------------
// apparie : manifeste, statut, secteurs
// ------------------------------------------------------------------------------------------

#[test]
fn apparie_manifeste_statut_et_secteurs() {
    let d = dossier_test("apparie");
    let run = d.join("run");
    let (rc, out, err) = kvu_dans(
        &d,
        &[
            "su2",
            "tube",
            "apparie",
            "--cotes",
            "4,4,2,2",
            "--repliques",
            "2",
            "--therm",
            "20",
            "--nmeas",
            "40",
            "--beta",
            "8,16",
            "--outdir",
            &s(&run),
        ],
    );
    assert_eq!(rc, 0, "{out}\n{err}");
    assert!(err.contains(AVERTISSEMENT), "{err}");
    let m = verifier_run_complet(&run);
    assert_eq!(m["mode"], "su2-tube-apparie");
    assert_eq!(m["parametres"]["statut"], "experimental", "{m}");
    assert!(
        m.get("geometrie").is_none(),
        "deux secteurs : pas de clé geometrie au niveau du run"
    );
    assert_eq!(m["geometrie_secteur0"]["cotes"], json!([4, 4, 2, 2]));
    assert_eq!(m["geometrie_secteur1"]["cotes"], json!([4, 4, 2, 2]));
    // MAJEURE 5 (relecture c16) : moyennes, erreurs de blocs et différence appariée RECONSTRUITES depuis les séries écrites
    // (V_bII_rRR_sS.txt, une valeur par mesure) puis comparées au JSON ; un signe de DV inversé ou une erreur nulle rougissent.
    let a = lire_json(&run.join("apparie.json"));
    let (nmeas, nrep, nb) = (40usize, 2usize, 20usize);
    let proche = |x: &Value, y: f64, quoi: &str| {
        let x = x.as_f64().unwrap_or_else(|| panic!("{quoi} absent : {a}"));
        assert!(
            (x - y).abs() <= 1e-12 * y.abs().max(1.0),
            "{quoi} : JSON {x}, reconstruit {y}"
        );
    };
    for (ib, beta) in [8.0f64, 16.0].into_iter().enumerate() {
        let r = &a["resultats"][ib];
        assert_eq!(r["beta"].as_f64(), Some(beta));
        let mut sect = [(0.0f64, 0.0f64); 2];
        for (sc, sm) in sect.iter_mut().enumerate() {
            let (mut ms, mut es) = (Vec::new(), Vec::new());
            for rep in 0..nrep {
                let f = run.join(format!("V_b{ib:02}_r{rep:02}_s{sc}.txt"));
                let x: Vec<f64> = std::fs::read_to_string(&f)
                    .unwrap_or_else(|e| panic!("{} : {e}", f.display()))
                    .lines()
                    .map(|l| l.trim().parse().unwrap())
                    .collect();
                assert_eq!(x.len(), nmeas, "{}", f.display());
                let moy = x.iter().sum::<f64>() / nmeas as f64;
                let t = nmeas / nb;
                let bm: Vec<f64> = (0..nb)
                    .map(|k| x[k * t..(k + 1) * t].iter().sum::<f64>() / t as f64)
                    .collect();
                let mb = bm.iter().sum::<f64>() / nb as f64;
                let err = (bm.iter().map(|y| (y - mb) * (y - mb)).sum::<f64>()
                    / (nb as f64 * (nb as f64 - 1.0)))
                    .sqrt();
                assert!(err > 0.0, "série dégénérée {}", f.display());
                let pr = &r["par_replique"][format!("V{sc}")][rep];
                proche(
                    &pr[0],
                    moy,
                    &format!("β={beta} V{sc} réplique {rep} : moyenne"),
                );
                proche(
                    &pr[1],
                    err,
                    &format!("β={beta} V{sc} réplique {rep} : erreur"),
                );
                ms.push(moy);
                es.push(err);
            }
            let n = nrep as f64;
            *sm = (
                ms.iter().sum::<f64>() / n,
                es.iter().map(|e| e * e).sum::<f64>().sqrt() / n,
            );
            proche(&r[format!("V{sc}")][0], sm.0, &format!("β={beta} V{sc}"));
            proche(
                &r[format!("V{sc}")][1],
                sm.1,
                &format!("β={beta} erreur V{sc}"),
            );
        }
        let dv = sect[1].0 - sect[0].0;
        let edv = (sect[0].1 * sect[0].1 + sect[1].1 * sect[1].1).sqrt();
        proche(&r["DV"][0], dv, &format!("β={beta} DV = V1 − V0"));
        proche(&r["DV"][1], edv, &format!("β={beta} erreur de DV"));
        assert!(edv > 0.0);
    }
}

// ------------------------------------------------------------------------------------------
// Graines (objection 4)
// ------------------------------------------------------------------------------------------

#[test]
fn graines_des_taches_sans_collision() {
    for nrep in [1usize, 4, 500, 501, 1200] {
        let mut vues = HashSet::new();
        for ib in 0..3 {
            for rep in 0..nrep {
                for sect in 0..2 {
                    assert!(
                        vues.insert(graine_tache(1, ib, rep, sect, nrep).expect("k représentable")),
                        "collision de graines : {nrep} répliques, (ib, rep, sect) = ({ib}, {rep}, {sect})"
                    );
                }
            }
        }
    }
    // jusqu'à 500 répliques, graines des versions précédentes inchangées : graine + (1000·ib + 2·rep + sect)·7919
    for (ib, rep, sect) in [(0usize, 0usize, 0usize), (1, 0, 0), (2, 499, 1), (5, 3, 1)] {
        assert_eq!(
            graine_tache(7, ib, rep, sect, 500),
            Some(7 + ((1000 * ib + 2 * rep + sect) as u64) * 7919)
        );
    }
    // MINEURE 4 (relecture c16) : k borné AVANT la multiplication. 2⁶⁰ répliques : (8, 0, 0) donnait la graine de (0, 0, 0)
    // (k = 8·2⁶¹ ≡ 0 modulo 2⁶⁴) ; désormais k non représentable ⇒ None (refus d'usage en amont).
    let n60 = 1usize << 60;
    assert!(graine_tache(1, 0, 0, 0, n60).is_some());
    assert_eq!(
        graine_tache(1, 8, 0, 0, n60),
        None,
        "k = 8·2⁶¹ déborde : refus attendu"
    );
    assert_eq!(
        graine_tache(1, 0, usize::MAX / 2, 1, usize::MAX / 2 + 1),
        None
    );
}

#[test]
fn apparie_refuse_des_cles_non_representables() {
    // MINEURE 4 : refus d'usage (code 2) avant toute écriture et tout calcul
    let d = dossier_test("refus_cles");
    let run = d.join("run");
    let args = [
        "su2",
        "tube",
        "apparie",
        "--cotes",
        "2,2,2,2",
        "--beta",
        "1,2",
        "--depart",
        "chaud",
        "--therm",
        "0",
        "--nmeas",
        "20",
        "--series",
        "non",
        "--repliques",
        "9223372036854775807",
        "--outdir",
        &s(&run),
    ];
    verifier_refus_rc(&d, &run, &args, &[], RC_USAGE, "repliques", 3);
}

// ------------------------------------------------------------------------------------------
// Refus d'usage (objections 3, 6, 7, 8, 9, 11) : code 2, rien d'écrit
// ------------------------------------------------------------------------------------------

#[test]
fn gen_cotes_refuse_les_options_ignorees() {
    // objection 6 : options de `su2 gen` sans effet sur le chemin du tube
    let d = dossier_test("refus_gen");
    let run = d.join("run");
    let r = s(&run);
    for (opt, val) in [
        ("--L", "8"),
        ("--options", "compat-prod"),
        ("--bc", "cold"),
        ("--bc-graine", "3"),
        ("--jauge", "5"),
        ("--pre", "10"),
        ("--bloc", "10"),
        ("--source-hash", "abc"),
    ] {
        let args = [
            "su2",
            "gen",
            "--cotes",
            "4,4,2,2",
            "--backend",
            "cpu",
            "--beta",
            "2.0",
            "--therm",
            "2",
            "--nmeas",
            "2",
            opt,
            val,
            "--outdir",
            &r,
        ];
        verifier_refus(&d, &run, &args, opt);
    }
}

#[test]
fn gen_cotes_refuse_la_periode_nulle() {
    // objection 8
    let d = dossier_test("refus_periode");
    let run = d.join("run");
    let args = [
        "su2",
        "gen",
        "--cotes",
        "4,4,2,2",
        "--backend",
        "cpu",
        "--beta",
        "2.0",
        "--therm",
        "2",
        "--nmeas",
        "2",
        "--mesures",
        "tube:01:1:1:0",
        "--outdir",
        &s(&run),
    ];
    verifier_refus(&d, &run, &args, "période");
}

#[test]
fn apparie_valide_ses_options_avant_tout_calcul() {
    // objections 7 (--repliques 0, --skip 0, --nmeas < 20) et 3 (--nmeas non multiple de 20)
    let d = dossier_test("refus_apparie");
    let run = d.join("run");
    let r = s(&run);
    for (opt, val) in [
        ("--repliques", "0"),
        ("--skip", "0"),
        ("--nmeas", "10"),
        ("--nmeas", "39"),
    ] {
        let mut args = vec![
            "su2", "tube", "apparie", "--cotes", "4,4,2,2", "--beta", "8", "--therm", "10",
            "--series", "non",
        ];
        if opt != "--nmeas" {
            args.extend_from_slice(&["--nmeas", "20"]);
        }
        args.extend_from_slice(&[opt, val, "--outdir", &r]);
        verifier_refus(&d, &run, &args, opt);
    }
}

#[test]
fn smin_cfg_un_geometrie_json_par_dossier() {
    // objection 9 : un geometrie.json décrit TOUTES les configurations de son dossier (notes geom §5-6)
    let d = dossier_test("smin_cfg");
    let a = d.join("a.npy");
    smin(&d, &["--cotes", "4,4,2,2", "--cfg", &s(&a)]);
    let gj = std::fs::read(d.join("geometrie.json")).unwrap();
    assert_eq!(
        lire_json(&d.join("geometrie.json"))["cotes"],
        json!([4, 4, 2, 2])
    );
    let fm = lire_json(&d.join("a.npy.manifest.json"));
    assert_eq!(fm["parametres"]["statut"], "experimental", "{fm}");
    assert_eq!(fm["geometrie"]["cotes"], json!([4, 4, 2, 2]));
    // même géométrie, même dossier : accepté
    smin(&d, &["--cotes", "4,4,2,2", "--cfg", &s(&d.join("a2.npy"))]);
    // autre géométrie dans le même dossier : refus, rien d'écrit, geometrie.json inchangé
    let b = d.join("b.npy");
    verifier_refus(
        &d,
        &b,
        &[
            "su2",
            "tube",
            "smin",
            "--cotes",
            "4,4,3,3",
            "--balayages",
            "200",
            "--cfg",
            &s(&b),
        ],
        "geometrie.json",
    );
    assert!(!d.join("b.npy.manifest.json").exists());
    assert_eq!(
        std::fs::read(d.join("geometrie.json")).unwrap(),
        gj,
        "geometrie.json réécrit"
    );
}

#[test]
fn reprise_dun_run_cotes_refusee_clairement() {
    // objection 11
    let d = dossier_test("reprise");
    let run = d.join("run");
    let (rc, _, err) = kvu_dans(
        &d,
        &[
            "su2",
            "gen",
            "--cotes",
            "4,4,2,2",
            "--backend",
            "cpu",
            "--beta",
            "2.0",
            "--therm",
            "2",
            "--nmeas",
            "2",
            "--outdir",
            &s(&run),
        ],
    );
    assert_eq!(rc, 0, "{err}");
    let avant = std::fs::read(run.join("manifest.json")).unwrap();
    let (rc, _, err) = kvu_dans(
        &d,
        &["su2", "gen", "--backend", "cpu", "--reprendre", &s(&run)],
    );
    assert_eq!(rc, RC_USAGE, "{err}");
    assert!(err.contains("non reprenable"), "{err}");
    assert_eq!(std::fs::read(run.join("manifest.json")).unwrap(), avant);
}

// ------------------------------------------------------------------------------------------
// Aide et messages (étiquette EXPÉRIMENTALE ; message de `su2` sans sous-commande)
// ------------------------------------------------------------------------------------------

#[test]
fn aide_marquee_experimentale() {
    let d = dossier_test("aide");
    for a in [
        vec!["su2", "tube", "--help"],
        vec!["su2", "tube", "smin", "--help"],
        vec!["su2", "tube", "apparie", "--help"],
    ] {
        let (rc, out, _) = kvu_dans(&d, &a);
        assert_eq!(rc, 0);
        assert!(out.starts_with("[EXPÉRIMENTAL]"), "{a:?} : {out}");
    }
    let (rc, out, _) = kvu_dans(&d, &["su2", "gen", "--help"]);
    assert_eq!(rc, 0);
    let i = out
        .find("--cotes <cotes>")
        .expect("--cotes absent de l'aide de su2 gen");
    assert!(
        out[i + "--cotes <cotes>".len()..]
            .trim_start()
            .starts_with("[EXPÉRIMENTAL]"),
        "{out}"
    );
}

#[test]
fn message_de_su2_sans_sous_commande_cite_tube() {
    let m = clap::Command::new("su2")
        .try_get_matches_from(["su2"])
        .unwrap();
    let e = kvu::su2::cli::executer("su2", &m).unwrap_err().to_string();
    assert!(e.contains("| tube |"), "{e}");
}

// ------------------------------------------------------------------------------------------
// Exemples de la note docs/v0.2/notes/tube.md, exécutés avec leurs arguments exacts
// ------------------------------------------------------------------------------------------

#[test]
fn exemples_de_la_note_tournent() {
    let note = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/v0.2/notes/tube.md");
    let t = std::fs::read_to_string(&note).unwrap_or_else(|e| panic!("{} : {e}", note.display()));
    let mut en_bloc = false;
    let mut cmds: Vec<&str> = Vec::new();
    for l in t.lines() {
        if l.trim_start().starts_with("```") {
            en_bloc = !en_bloc;
        } else if en_bloc && l.starts_with("kvu ") {
            cmds.push(l);
        }
    }
    assert_eq!(
        cmds.len(),
        3,
        "trois exemples attendus dans la note : {cmds:?}"
    );
    let d = dossier_test("note");
    for c in &cmds {
        let args: Vec<&str> = c.split_whitespace().skip(1).collect();
        let (rc, out, err) = kvu_dans(&d, &args);
        assert_eq!(rc, 0, "« {c} » : {out}\n{err}");
        assert!(err.contains(AVERTISSEMENT), "« {c} » : {err}");
        if args[1] == "tube" && args[2] == "smin" {
            let j: Value = serde_json::from_str(out.lines().last().unwrap()).unwrap();
            assert!(
                (j["S_sur_beta"].as_f64().unwrap() - SMIN_4_2).abs() < 1e-8,
                "{j}"
            );
        }
    }
    verifier_run_complet(&d.join("tube_gen"));
    verifier_run_complet(&d.join("tube_apparie"));
    assert!(
        d.join("tube_gen/tube.csv").exists()
            && d.join("smin.npy").exists()
            && d.join("geometrie.json").exists()
    );
}

// ------------------------------------------------------------------------------------------
// Relecture c16 (Sol), BLOQUANTES 1 et 2 : tailles et totaux vérifiés, bornés avant toute écriture
// ------------------------------------------------------------------------------------------

#[test]
fn balayages_par_chaine_bornes_au_compteur_philox() {
    use kvu::su2::tube::cli::balayages_par_chaine;
    let deux32 = 1u64 << 32;
    assert_eq!(balayages_par_chaine(deux32 - 2, 1, 1).unwrap(), deux32 - 1);
    assert_eq!(balayages_par_chaine(10, 20, 3).unwrap(), 70);
    for (th, nm, sk) in [
        (deux32 - 1, 1, 1),
        (deux32, 1, 1),
        (u64::MAX, 1, 1),
        (0, u64::MAX, 2),
        (0, 3, 6_148_914_691_236_517_206),
        (5, deux32, 1),
    ] {
        let e = balayages_par_chaine(th, nm, sk).expect_err(&format!("({th}, {nm}, {sk}) accepté"));
        assert_eq!(e.code_sortie(), RC_USAGE);
    }
}

#[test]
fn apparie_tailles_extremes_refusees_sans_panique() {
    // BLOQUANTE 1 : --nmeas 18446744073709551600 (multiple de 20) faisait paniquer Vec::with_capacity
    let d = dossier_test("extremes_apparie");
    let run = d.join("run");
    let r = s(&run);
    for (therm, nmeas, skip) in [
        ("0", "18446744073709551600", "1"),
        ("0", "4294967300", "1"),
        ("0", "20", "214748365"),
        ("18446744073709551615", "20", "1"),
    ] {
        let args = [
            "su2", "tube", "apparie", "--cotes", "2,2,2,2", "--beta", "1", "--depart", "chaud",
            "--therm", therm, "--nmeas", nmeas, "--skip", skip, "--outdir", &r,
        ];
        verifier_refus_rc(&d, &run, &args, &[], RC_USAGE, "2³²", 10);
    }
}

#[test]
fn gen_cotes_totaux_refuses_avant_ecriture() {
    // BLOQUANTE 2 : au-delà de 2³² − 1 balayages par chaîne, le compteur Philox (sweep_id, 32 bits) se répéterait ;
    // therm + nmeas·skip débordait après l'ouverture du manifeste
    let d = dossier_test("totaux_gen");
    let run = d.join("run");
    let r = s(&run);
    for (therm, nmeas, skip) in [
        ("4294967296", "1", "1"),
        ("4294967295", "1", "1"),
        ("18446744073709551615", "1", "1"),
        ("0", "3", "6148914691236517206"),
    ] {
        let args = [
            "su2",
            "gen",
            "--cotes",
            "2,2,2,2",
            "--backend",
            "cpu",
            "--beta",
            "2.0",
            "--nor",
            "0",
            "--therm",
            therm,
            "--nmeas",
            nmeas,
            "--skip",
            skip,
            "--outdir",
            &r,
        ];
        verifier_refus_rc(&d, &run, &args, &[], RC_USAGE, "2³²", 10);
    }
}

#[test]
fn gen_cotes_lots_periodes_et_graines_valides() {
    // tailles dépendant d'une entrée : nombre de chaînes, période des observables, nombre de mesures ; graines distinctes
    let d = dossier_test("lots_gen");
    let run = d.join("run");
    let r = s(&run);
    let base = [
        "su2",
        "gen",
        "--cotes",
        "2,2,2,2",
        "--backend",
        "cpu",
        "--therm",
        "2",
        "--outdir",
        &r,
    ];
    let cas: [(&[&str], &str); 7] = [
        (
            &[
                "--beta",
                "2",
                "--graine",
                "1",
                "--graine-pas",
                "1",
                "--chaines",
                "18446744073709551615",
                "--nmeas",
                "2",
            ],
            "--chaines",
        ),
        (&["--beta", "2", "--graine", "1", "--nmeas", "0"], "--nmeas"),
        (
            &[
                "--beta",
                "2",
                "--graine",
                "1",
                "--nmeas",
                "20",
                "--mesures",
                "tube:01:1:1:30",
            ],
            "période",
        ),
        (
            &[
                "--beta",
                "2",
                "--graine",
                "1",
                "--nmeas",
                "20",
                "--mesures",
                "tube:01:1:1:15",
            ],
            "période",
        ),
        (
            &["--beta", "2.2,2.5", "--graine", "7,7", "--nmeas", "2"],
            "graines",
        ),
        (
            &[
                "--beta",
                "2",
                "--graine",
                "1",
                "--graine-pas",
                "0",
                "--chaines",
                "2",
                "--nmeas",
                "2",
            ],
            "graines",
        ),
        (
            &[
                "--beta",
                "2",
                "--graine",
                "1",
                "--graine-pas",
                "9223372036854775808",
                "--chaines",
                "3",
                "--nmeas",
                "2",
            ],
            "graines",
        ),
    ];
    for (extra, fragment) in cas {
        let mut args: Vec<&str> = base.to_vec();
        args.extend_from_slice(extra);
        verifier_refus_rc(&d, &run, &args, &[], RC_USAGE, fragment, 10);
    }
}

#[test]
fn admission_memoire_avant_toute_ecriture() {
    // réservations dépendant des entrées (côtés, chaînes, séries, R et T) : admission mémoire de kvu (rc 5) avant toute
    // écriture ; ici forcée par un plancher de RAM démesuré
    let d = dossier_test("admission");
    let env = [("KEVINOTRON_RAM_MIN_GO", "1000000000")];
    let run = d.join("gen");
    let args = [
        "su2",
        "gen",
        "--cotes",
        "4,4,2,2",
        "--backend",
        "cpu",
        "--beta",
        "2",
        "--therm",
        "2",
        "--nmeas",
        "2",
        "--outdir",
        &s(&run),
    ];
    verifier_refus_rc(&d, &run, &args, &env, 5, "ressources", 30);
    let run = d.join("apparie");
    let args = [
        "su2",
        "tube",
        "apparie",
        "--cotes",
        "4,4,2,2",
        "--beta",
        "8",
        "--therm",
        "2",
        "--nmeas",
        "20",
        "--outdir",
        &s(&run),
    ];
    verifier_refus_rc(&d, &run, &args, &env, 5, "ressources", 30);
    let cfg = d.join("s.npy");
    let args = [
        "su2",
        "tube",
        "smin",
        "--cotes",
        "4,4,2,2",
        "--cfg",
        &s(&cfg),
    ];
    verifier_refus_rc(&d, &cfg, &args, &env, 5, "ressources", 30);
}

#[test]
fn projection_disque_avant_toute_ecriture() {
    // 10⁶ configurations de 32⁴ (256 Mio chacune, ~256 Tio) : mémoire admise (~2,6 Gio, catalogue compris), écriture projetée
    // refusée par la garde SSD (rc 7) à l'ouverture, avant toute écriture
    let d = dossier_test("projection");
    let run = d.join("run");
    let args = [
        "su2",
        "gen",
        "--cotes",
        "32,32,32,32",
        "--backend",
        "cpu",
        "--beta",
        "2",
        "--therm",
        "0",
        "--nor",
        "0",
        "--nmeas",
        "1000000",
        "--dump-every",
        "1",
        "--outdir",
        &s(&run),
    ];
    verifier_refus_rc(&d, &run, &args, &[], 7, "SSD", 5);
}

// ------------------------------------------------------------------------------------------
// Relecture c16, MAJEURE 3 : smin (entrées, convergence, « candidat minimum »)
// ------------------------------------------------------------------------------------------

#[test]
fn smin_entrees_refusees() {
    let d = dossier_test("smin_refus");
    let cfg = d.join("s.npy");
    let c = s(&cfg);
    let cas: [(&[&str], &str); 8] = [
        (&["--gtol", "NaN"], "--gtol"),
        (&["--gtol", "0"], "--gtol"),
        (&["--gtol=-1"], "--gtol"),
        (&["--bruit", "NaN"], "--bruit"),
        (&["--bruit", "0"], "--bruit"),
        (&["--bruit", "0", "--depart", "abelien"], "--bruit"),
        (&["--bruit", "7"], "--bruit"),
        (&["--balayages", "18446744073709551615"], "--balayages"),
    ];
    for (extra, fragment) in cas {
        let mut args = vec!["su2", "tube", "smin", "--cotes", "4,4,2,2", "--cfg", &c];
        args.extend_from_slice(extra);
        verifier_refus_rc(&d, &cfg, &args, &[], RC_USAGE, fragment, 10);
    }
    // bruit nul admis depuis un départ chaud (non stationnaire)
    let (j, _) = smin(
        &d,
        &[
            "--cotes",
            "4,4,2,2",
            "--depart",
            "chaud",
            "--bruit",
            "0",
            "--balayages",
            "200",
        ],
    );
    assert!(j["S_sur_beta"].as_f64().is_some_and(f64::is_finite), "{j}");
}

#[test]
fn smin_convergence_explicite_et_candidat_minimum() {
    let d = dossier_test("smin_conv");
    let (j, err) = smin(&d, &["--cotes", "4,4,2,2"]);
    let cv = &j["convergence"];
    assert_eq!(cv["atteinte"], true, "{j}");
    assert!(
        cv["grad_max"].as_f64().unwrap() < cv["gtol"].as_f64().unwrap(),
        "{j}"
    );
    assert!(
        j["nature"]
            .as_str()
            .unwrap_or("")
            .contains("candidat minimum"),
        "{j}"
    );
    assert!(!err.contains("NON atteinte"), "{err}");
    // aucun balayage depuis un départ chaud : convergence NON atteinte, dite (dans le domaine admis)
    let (j, err) = smin(
        &d,
        &[
            "--cotes",
            "4,4,2,2",
            "--depart",
            "chaud",
            "--balayages",
            "0",
        ],
    );
    assert_eq!(j["convergence"]["atteinte"], false, "{j}");
    assert!(
        j["convergence"]["grad_max"].as_f64().unwrap() >= 1e-10,
        "{j}"
    );
    assert!(err.contains("NON atteinte"), "{err}");
    for a in [
        vec!["su2", "tube", "smin", "--help"],
        vec!["su2", "tube", "apparie", "--help"],
    ] {
        let (rc, out, _) = kvu_dans(&d, &a);
        assert_eq!(rc, 0);
        assert!(out.contains("candidat minimum"), "{a:?} : {out}");
    }
    let run = d.join("apparie");
    let (rc, _, err) = kvu_dans(
        &d,
        &[
            "su2",
            "tube",
            "apparie",
            "--cotes",
            "4,4,2,2",
            "--beta",
            "8",
            "--therm",
            "2",
            "--nmeas",
            "20",
            "--series",
            "non",
            "--outdir",
            &s(&run),
        ],
    );
    assert_eq!(rc, 0, "{err}");
    let a = lire_json(&run.join("apparie.json"));
    assert_eq!(
        a["candidat_minimum_secteur1"]["convergence_atteinte"], true,
        "{a}"
    );
    assert!(a.get("minimiseur_secteur1").is_none(), "{a}");
}

// ------------------------------------------------------------------------------------------
// Relecture c16, MAJEURE 7 et MINEURE 8 : scripts/tube/ti_etalonnage.py
// ------------------------------------------------------------------------------------------

fn python_numpy() -> bool {
    let ok = std::process::Command::new("python3")
        .args(["-I", "-c", "import numpy"])
        .output()
        .is_ok_and(|o| o.status.success());
    if !ok {
        eprintln!("SKIP : python3 ou numpy absent");
    }
    ok
}

/// apparie.json factice (grille β = 0,25 … 1,0) pour l'étalonnage : seuls les champs lus par ti_etalonnage.py.
fn apparie_factice(dir: &Path, cotes: [usize; 4], secteur0: &str, secteur1: &str) {
    std::fs::create_dir_all(dir).unwrap();
    let vol: usize = cotes.iter().product();
    let res: Vec<Value> = [0.25f64, 0.5, 0.75, 1.0]
        .iter()
        .map(|&b| {
            let v0 = 6.0 * vol as f64 * (1.0 - 0.1 * b);
            json!({"beta": b, "V0": [v0, 0.01], "V1": [v0 + 0.02 * b, 0.01], "DV": [0.02 * b, 0.014],
                   "par_replique": {"V0": [[v0, 0.01], [v0, 0.01]], "V1": [[v0, 0.01], [v0, 0.01]]}})
        })
        .collect();
    let j = json!({"cotes": cotes, "secteur0": secteur0, "secteur1": secteur1, "resultats": res});
    std::fs::write(
        dir.join("apparie.json"),
        serde_json::to_vec_pretty(&j).unwrap(),
    )
    .unwrap();
}

/// Lance ti_etalonnage.py ; sa sortie est du JSON de Python (NaN pour les quadratures au pas double non définies), lu ici
/// avec NaN → null.
fn etalonnage(d: &Path, dirs: &[PathBuf], extra: &[&str]) -> (i32, String, String) {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/tube/ti_etalonnage.py");
    let mut c = std::process::Command::new("python3");
    c.arg("-I").arg(&script).current_dir(d);
    for x in dirs {
        c.arg(x);
    }
    let o = c.args(extra).output().unwrap();
    (
        o.status.code().unwrap_or(RC_SIGNAL),
        String::from_utf8_lossy(&o.stdout).replace("NaN", "null"),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

#[test]
fn etalonnage_secteurs_hors_protocole_refuses_et_p4_algebrique() {
    if !python_numpy() {
        return;
    }
    let d = dossier_test("etalonnage");
    let cotes = [
        [2, 2, 2, 2],
        [4, 2, 2, 2],
        [4, 4, 2, 2],
        [4, 4, 4, 2],
        [4, 4, 4, 4],
    ];
    // hors protocole : fibre 01 tordue dans les deux secteurs, plan ajouté 23 ; l'ancien script prenait « 01 » (premier plan)
    let hors: Vec<PathBuf> = cotes
        .iter()
        .enumerate()
        .map(|(j, c)| {
            let x = d.join(format!("hors{j}"));
            apparie_factice(&x, *c, "01", "01,23");
            x
        })
        .collect();
    let (rc, _, err) = etalonnage(&d, &hors, &[]);
    assert_ne!(rc, 0, "secteurs hors protocole acceptés : {err}");
    assert!(err.contains("hors protocole"), "{err}");
    // protocole (fibre aucune, un plan long tordu) : référence calculée sur le plan AJOUTÉ (ici 23 : A = ℓ2·ℓ3)
    let bons: Vec<PathBuf> = cotes
        .iter()
        .enumerate()
        .map(|(j, c)| {
            let x = d.join(format!("bon{j}"));
            apparie_factice(&x, *c, "aucune", "23");
            x
        })
        .collect();
    let (rc, out, err) = etalonnage(&d, &bons, &[]);
    assert_eq!(rc, 0, "{err}");
    let j: Value = serde_json::from_str(&out).unwrap();
    let pt = &j["points"][1];
    let t = pt["t"].as_f64().unwrap();
    for (k, c) in cotes.iter().enumerate() {
        let (vol, a) = ((c[0] * c[1] * c[2] * c[3]) as f64, (c[2] * c[3]) as i32);
        let attendu = -2.0 * vol / a as f64 * t.powi(a);
        let lu = pt["logR_SC"][k].as_f64().unwrap();
        assert!(
            (lu - attendu).abs() <= 1e-12 * attendu.abs(),
            "G{k} : {lu} ≠ {attendu}"
        );
    }
    // MINEURE 8 : P4 est une identité algébrique (pas une validation) ; le mutant n'est jugé que par des références indépendantes
    assert!(
        j["P4_nature"].as_str().unwrap_or("").contains("algébrique"),
        "{j}"
    );
    let noms: Vec<&str> = j["verdicts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v[0].as_str().unwrap())
        .collect();
    assert!(
        noms.contains(&"P4_identite_algebrique") && !noms.contains(&"P4"),
        "{noms:?}"
    );
    let (rc, out, err) = etalonnage(&d, &bons, &["--mutant-decalage"]);
    assert_eq!(rc, 0, "{err}");
    let j: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(j["mutant_juge_par"], json!(["P1", "P1_abs", "P2"]), "{j}");
    assert!(j["mutant_rouge"].is_boolean(), "{j}");
    // (MINEURE 18 : la fixture ci-dessus n'est pas conçue pour rejeter le mutant ; voir le test suivant, qui exige `true`)
}

// ------------------------------------------------------------------------------------------
// Propriété (demande du pilote, 11/11 soir) : deux flux censés être indépendants ne partagent jamais leurs nombres aléatoires
// ------------------------------------------------------------------------------------------

#[test]
fn flux_independants_sans_nombres_aleatoires_partages() {
    use kvu::commun::{ETIQUETTE_HB, ETIQUETTE_INIT};
    use kvu::su2::tube::cli::{
        chaines_tube, taches_apparie, BETAS_MAX, CHAINES_MAX, REPLIQUES_MAX,
    };
    use kvu::su2::tube::reseau::cle_philox;
    // graines : 0, 1 (défaut), u64::MAX et 1000 graines tirées (SplitMix64, état initial fixe)
    let mut graines = vec![0u64, 1, u64::MAX];
    let mut x = 0x2026_1111_u64;
    for _ in 0..1000 {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        graines.push(z ^ (z >> 31));
    }
    // compteur de départ : le même pour tous les flux d'un run (balayage 0, tirage 0 ; étiquettes du bain et du départ chaud
    // distinctes) ; la clé est celle de cle_philox, que ChaineTube::nouvelle appelle
    assert_ne!(ETIQUETTE_HB, ETIQUETTE_INIT);
    let g0 = 0xDEAD_BEEF_0123_4567_u64;
    let c = ChaineTube::nouvelle(GeoA::nouvelle([2, 2, 2, 2]).unwrap(), g0);
    assert_eq!(
        (c.k0, c.k1, c.sweep_id),
        (cle_philox(g0).0, cle_philox(g0).1, 0)
    );
    let depart = |graine: u64| (cle_philox(graine), 0u32, 0u32);
    // apparie : toutes les tâches β × réplique × secteur, telles que le run les construit ; plus grand lot admis
    // (BETAS_MAX × REPLIQUES_MAX) pour 0, 1 et u64::MAX, 3 β × REPLIQUES_MAX pour les 1000 graines tirées
    for (i, &g) in graines.iter().enumerate() {
        let nbeta = if i < 3 { BETAS_MAX } else { 3 };
        let t = taches_apparie(g, nbeta, REPLIQUES_MAX).unwrap();
        assert_eq!(t.len(), 2 * nbeta * REPLIQUES_MAX);
        let mut vus = HashSet::with_capacity(t.len());
        for &(ib, rep, sect, s) in &t {
            assert!(
                vus.insert(depart(s)),
                "apparie, graine {g} : la tâche (ib, rep, sect) = ({ib}, {rep}, {sect}) partage clé et compteur de départ"
            );
        }
    }
    assert!(taches_apparie(1, 3, REPLIQUES_MAX + 1).is_err());
    assert!(taches_apparie(1, BETAS_MAX + 1, 1).is_err());
    // gen --cotes : lot graine + i·pas (pas 1 et 7919) de CHAINES_MAX chaînes pour 0, 1 et u64::MAX, de 64 pour les autres
    for (i, &g) in graines.iter().enumerate() {
        let n = if i < 3 { CHAINES_MAX } else { 64 };
        for pas in [1u64, 7919] {
            let ch = chaines_tube(&[2.3], &[g], Some(n), Some(pas)).unwrap();
            assert_eq!(ch.len(), n);
            let mut vus = HashSet::with_capacity(n);
            for (j, c) in ch.iter().enumerate() {
                assert!(
                    vus.insert(depart(c.graine.0)),
                    "gen, graine {g}, pas {pas} : chaîne {j}"
                );
            }
        }
    }
    // collisions forcées : refusées (pas = 2⁶³ : la 3ᵉ chaîne retombe sur la 1ʳᵉ ; pas = 0 ; graine répétée ; lot trop grand)
    assert!(chaines_tube(&[2.3], &[1], Some(3), Some(1 << 63)).is_err());
    assert!(chaines_tube(&[2.3], &[1], Some(2), Some(0)).is_err());
    assert!(chaines_tube(&[2.2, 2.5], &[7, 7], None, None).is_err());
    assert!(chaines_tube(&[2.3], &[1], Some(CHAINES_MAX + 1), Some(1)).is_err());
}

// ------------------------------------------------------------------------------------------
// Révision 3 (relecture c16b) : DOMAINE ADMIS, validé en un seul endroit, parcouru par un test paramétré
// ------------------------------------------------------------------------------------------

/// Ligne de commande VALIDE de `commande` (`gen` = `su2 gen --cotes`, `smin`, `apparie`) où `option` reçoit `valeur` et où
/// les autres options sont à des valeurs qui laissent `option` atteindre sa borne (totaux et contraintes croisées tenus).
fn argv_domaine(commande: &str, option: &str, valeur: &str) -> Vec<String> {
    let v = valeur.to_string();
    let mut o: Vec<(&str, String)> = match commande {
        "gen" => vec![
            ("--cotes", "4,4,2,2".into()),
            ("--beta", "2".into()),
            ("--graine", "1".into()),
            ("--therm", "0".into()),
            ("--nmeas", "1".into()),
            ("--skip", "1".into()),
            ("--nor", "1".into()),
        ],
        "smin" => vec![
            ("--cotes", "4,4,2,2".into()),
            ("--graine", "1".into()),
            ("--gtol", "1e-10".into()),
            ("--bruit", "1e-3".into()),
            ("--balayages", "200".into()),
        ],
        _ => vec![
            ("--cotes", "4,4,2,2".into()),
            ("--beta", "8".into()),
            ("--graine", "1".into()),
            ("--therm", "0".into()),
            ("--nmeas", "20".into()),
            ("--skip", "1".into()),
            ("--nor", "1".into()),
            ("--repliques", "1".into()),
        ],
    };
    let mut fixer = |k: &'static str, val: String| match o.iter_mut().find(|x| x.0 == k) {
        Some(x) => x.1 = val,
        None => o.push((k, val)),
    };
    match option {
        "volume" => fixer("--cotes", v),
        "--cotes" => fixer("--cotes", format!("{v},4,2,2")),
        "nombre de β" => fixer("--beta", v),
        "--chaines" => {
            fixer("--chaines", v);
            fixer("--graine-pas", "1".into());
        }
        "--dump-every" => {
            fixer("--nmeas", "4294967295".into());
            fixer("--dump-every", v);
        }
        "--mesures tube:…:N (période)" => {
            fixer("--nmeas", "4294967295".into());
            fixer("--mesures", format!("tube:01:1:1:{v}"));
        }
        "--mesures tube:ab:RMAX" => {
            fixer("--nmeas", "2".into());
            fixer("--cotes", "4096,1024,2,2".into());
            fixer("--mesures", format!("tube:01:{v}:1"));
        }
        "--mesures tube:ab:RMAX:TMAX" => {
            fixer("--nmeas", "2".into());
            fixer("--cotes", "1024,4096,2,2".into());
            fixer("--mesures", format!("tube:01:1:{v}"));
        }
        "--bruit" => {
            fixer("--depart", "chaud".into());
            fixer("--bruit", v);
        }
        autre => {
            let k: &'static str = match autre {
                "--beta" => "--beta",
                "--therm" => "--therm",
                "--nmeas" => "--nmeas",
                "--skip" => "--skip",
                "--nor" => "--nor",
                "--graine" => "--graine",
                "--graine-pas" => "--graine-pas",
                "--gtol" => "--gtol",
                "--balayages" => "--balayages",
                "--repliques" => "--repliques",
                x => panic!("option du domaine sans construction de test : {x}"),
            };
            fixer(k, v);
        }
    }
    let mut a: Vec<String> = match commande {
        "gen" => ["su2", "gen", "--backend", "cpu", "--outdir", "d_inexistant"]
            .map(String::from)
            .to_vec(),
        "smin" => ["su2", "tube", "smin"].map(String::from).to_vec(),
        _ => [
            "su2",
            "tube",
            "apparie",
            "--series",
            "non",
            "--outdir",
            "d_inexistant",
        ]
        .map(String::from)
        .to_vec(),
    };
    for (k, val) in o {
        a.push(format!("{k}={val}"));
    }
    a
}

/// Cas d'une ligne du domaine : (texte de la valeur, admis ?) — bornes, juste dehors, NaN, infinis, zéro, maximum du type.
fn cas_domaine(b: &kvu::su2::tube::domaine::Borne) -> Vec<(String, bool)> {
    let dedans = |x: f64| (b.min..=b.max).contains(&x);
    match b.option {
        "volume" => vec![
            ("2,2,2,2".into(), true),
            ("4096,1024,2,2".into(), true),
            ("4096,1025,2,2".into(), false),
        ],
        "nombre de β" => vec![
            ("8".into(), true),
            (vec!["8"; 256].join(","), true),
            (vec!["8"; 257].join(","), false),
            (String::new(), false),
        ],
        _ if b.entier => {
            // bornes entières exactes ; 2⁶⁴ − 1 (graines) n'est pas représentable en f64 : la borne vaut alors u64::MAX
            let min = b.min as u128;
            let max = if b.max >= u64::MAX as f64 {
                u64::MAX as u128
            } else {
                b.max as u128
            };
            let mut v = vec![
                (min.to_string(), true),
                (max.to_string(), true),
                (
                    if min == 0 {
                        "-1".to_string()
                    } else {
                        (min - 1).to_string()
                    },
                    false,
                ),
                ((max + 1).to_string(), false),
                ("NaN".into(), false),
                ("inf".into(), false),
                ("1.5".into(), false),
                ("0".into(), dedans(0.0)),
                (u64::MAX.to_string(), dedans(u64::MAX as f64)),
            ];
            if b.option == "--nmeas" && b.min == 20.0 {
                v.push(("30".into(), false)); // apparie : multiple de 20
            }
            v
        }
        _ => {
            let bas = if b.min == 0.0 {
                -1e-300
            } else {
                b.min * (1.0 - 1e-9)
            };
            vec![
                (format!("{}", b.min), true),
                (format!("{}", b.max), true),
                (format!("{bas}"), false),
                (format!("{}", b.max * (1.0 + 1e-9)), false),
                ("NaN".into(), false),
                ("inf".into(), false),
                ("-inf".into(), false),
                ("0".into(), dedans(0.0)),
                (format!("{}", f64::MAX), false),
            ]
        }
    }
}

#[test]
fn domaine_admis_parcouru_borne_par_borne() {
    use kvu::su2::tube::cli::analyser_seulement;
    use kvu::su2::tube::domaine::DOMAINE;
    let mut n = 0;
    for b in DOMAINE {
        for &commande in b.commandes {
            for (valeur, admis) in cas_domaine(b) {
                let argv = argv_domaine(commande, b.option, &valeur);
                let a: Vec<&str> = argv.iter().map(String::as_str).collect();
                let r = analyser_seulement(&a);
                match (admis, r) {
                    (true, Ok(())) => {}
                    (false, Err(e)) => assert_eq!(e.code_sortie(), RC_USAGE, "{a:?} : {e}"),
                    (true, Err(e)) => {
                        panic!("{commande} {} = {valeur} : refusé à tort : {e}", b.option)
                    }
                    (false, Ok(())) => panic!(
                        "{commande} {} = {valeur} : admis hors du domaine ({a:?})",
                        b.option
                    ),
                }
                n += 1;
            }
        }
    }
    assert!(n > 150, "{n} cas seulement");
}

#[test]
fn domaine_dans_l_aide_et_la_note() {
    use kvu::su2::tube::domaine::DOMAINE;
    let d = dossier_test("domaine_aide");
    let (rc, aide, _) = kvu_dans(&d, &["su2", "tube", "--help"]);
    assert_eq!(rc, 0);
    let note = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/v0.2/notes/tube.md"),
    )
    .unwrap();
    for b in DOMAINE {
        assert!(
            aide.contains(b.intervalle),
            "aide sans « {} » ({})",
            b.intervalle,
            b.option
        );
        assert!(
            note.contains(b.intervalle),
            "note sans « {} » ({})",
            b.intervalle,
            b.option
        );
    }
}

#[test]
fn contre_exemples_c16b_refuses_sans_ecrire() {
    let d = dossier_test("c16b");
    let run = d.join("run");
    let r = s(&run);
    // 10 : β = 1e308 (V₀ = 8, β·8 déborde : équipartition infinie rendue avec le code 0)
    let a10 = [
        "su2", "tube", "apparie", "--cotes", "2,2,2,2", "--beta", "1e308", "--fibre", "01",
        "--long", "23", "--depart", "minimum", "--therm", "0", "--nmeas", "20", "--nor", "0",
        "--series", "non", "--outdir", &r,
    ];
    verifier_refus_rc(&d, &run, &a10, &[], RC_USAGE, "--beta", 10);
    // 11 : β = 1e-300 (exp(−2α) = 1 : chaque lien épuise ses 10⁶ essais ; le cumul des essais débordait)
    let a11 = [
        "su2",
        "gen",
        "--cotes",
        "32,32,32,32",
        "--backend",
        "cpu",
        "--start",
        "cold",
        "--beta",
        "1e-300",
        "--therm",
        "0",
        "--nmeas",
        "1",
        "--skip",
        "4400000",
        "--nor",
        "0",
        "--outdir",
        &r,
    ];
    verifier_refus_rc(&d, &run, &a11, &[], RC_USAGE, "--beta", 10);
}

#[test]
fn smin_balayages_plafond_exact() {
    // 19 : --balayages 1 faisait un paquet entier de 200 balayages
    let d = dossier_test("smin_plafond");
    let (j, _) = smin(
        &d,
        &[
            "--cotes",
            "4,4,2,2",
            "--depart",
            "chaud",
            "--balayages",
            "1",
        ],
    );
    assert_eq!(j["balayages"], 1, "{j}");
    assert_eq!(j["convergence"]["atteinte"], false, "{j}");
}

#[test]
fn cumul_des_essais_sature() {
    // 11 (partie module) : le cumul des essais du bain thermique ne déborde pas
    let g = GeoA::nouvelle([2, 2, 2, 2]).unwrap();
    let pile = Pile::depuis_torsion(&g, &Torsion::analyser("aucune").unwrap());
    let mut c = ChaineTube::nouvelle(g, 3);
    c.init(true);
    c.n_tries = u64::MAX - 1;
    c.sweep_hb(2.3, &pile);
    assert_eq!(c.n_tries, u64::MAX, "cumul des essais non saturant");
}

#[test]
fn majorant_memoire_gen_couvre_series_et_catalogue() {
    use kvu::su2::tube::cli::majorant_memoire_gen;
    // 13 : côtés 2⁴, 67 108 865 mesures d'observables (R = T = 1). Séries à capacité exacte : par mesure V, fraction,
    // 6 plaquettes, 12 Polyakov (160 o), vecteur des boucles (en-tête 24 o, 8·R·T o, 16 o d'allocation) ; copie finale d'une
    // composante (8 o)
    let nobs = 67_108_865u64;
    let m = majorant_memoire_gen([2, 2, 2, 2], 1, nobs, 1, 1, 1, 0, false).expect("représentable");
    let series = nobs * (160 + 24 + 8 + 16 + 8);
    assert!(m >= series, "majorant {m} < séries {series}");
    // catalogue des configurations écrites (manifeste, liste des fichiers) : au moins 2 Kio par configuration
    let m2 =
        majorant_memoire_gen([2, 2, 2, 2], 1, 1_000_000, 0, 0, 0, 1, false).expect("représentable");
    assert!(
        m2 >= 1_000_000 * 2048,
        "catalogue de 10⁶ configurations non budgété : {m2}"
    );
}

#[test]
fn depart_minimum_ne_promet_s_nul_que_la_ou_c_est_vrai() {
    // 16 : avec --fibre 01, le secteur 0 part de l'identité (S/β = 8 sur 4,4,2,2), pas d'un vide d'action nulle
    let d = dossier_test("depart_s0");
    let (rc, aide, _) = kvu_dans(&d, &["su2", "tube", "apparie", "--help"]);
    assert_eq!(rc, 0);
    assert!(aide.contains("23 ou aucune"), "{aide}");
    for (fibre, attendu) in [("23", "vide"), ("01", "identité")] {
        let run = d.join(format!("r{fibre}"));
        let (rc, _, err) = kvu_dans(
            &d,
            &[
                "su2",
                "tube",
                "apparie",
                "--cotes",
                "4,4,2,2",
                "--fibre",
                fibre,
                "--long",
                if fibre == "23" { "01" } else { "23" },
                "--beta",
                "8",
                "--therm",
                "2",
                "--nmeas",
                "20",
                "--series",
                "non",
                "--outdir",
                &s(&run),
            ],
        );
        assert_eq!(rc, 0, "{err}");
        let a = lire_json(&run.join("apparie.json"));
        assert!(
            a["depart_secteur0"]
                .as_str()
                .unwrap_or("")
                .starts_with(attendu),
            "{fibre} : {a}"
        );
    }
}

#[test]
fn depart_secteur0_consigne_l_action_reelle_fibre_composite() {
    // 20 : --fibre 01,23 --long 02 sur 4,4,2,2 : le secteur 0 reçoit le vide U*(+,+), d'action S/β = 8 (quatre plaquettes de
    // la pile 01, chacune 2), PAS S = 0 ; l'action initiale réelle est consignée, calculée sur la configuration de départ
    let d = dossier_test("depart_s0_composite");
    for (fibre, long, nulle, attendue) in [
        ("01,23", "02", false, 8.0),
        ("01", "23", false, 8.0),
        ("23", "01", true, 0.0),
    ] {
        let run = d.join(format!("r{fibre}"));
        let (rc, _, err) = kvu_dans(
            &d,
            &[
                "su2",
                "tube",
                "apparie",
                "--cotes",
                "4,4,2,2",
                "--fibre",
                fibre,
                "--long",
                long,
                "--beta",
                "8",
                "--depart",
                "minimum",
                "--therm",
                "0",
                "--nmeas",
                "20",
                "--nor",
                "0",
                "--series",
                "non",
                "--outdir",
                &s(&run),
            ],
        );
        assert_eq!(rc, 0, "{fibre} : {err}");
        let a = lire_json(&run.join("apparie.json"));
        let x = a["action_initiale_secteur0"]
            .as_f64()
            .unwrap_or_else(|| panic!("{fibre} : action initiale non consignée : {a}"));
        assert!((x - attendue).abs() < 1e-9, "{fibre} : S/β = {x} : {a}");
        let texte = a["depart_secteur0"].as_str().unwrap_or("");
        assert_eq!(texte.contains("S = 0"), nulle, "{fibre} : {texte}");
    }
}

#[test]
fn majorant_memoire_gen_compte_les_sommes_partielles_de_wilson() {
    use kvu::su2::tube::cli::majorant_memoire_gen;
    // 21 : 768,768,2,2 (V = 2 359 296), R = T = 384 : les 576 sommes partielles simultanées (⌈V/4096⌉ = 576 blocs de
    // 2·R·T = 294 912 réels) pèsent 16·R·T·576 o = 1,2656 Gio de plus que les lignes de liens ((R+T)·32 + 48 o par site)
    let v = 768u64 * 768 * 2 * 2;
    let lignes = (384u64 + 384) * 32 + 48;
    let partielles = 16u64 * 384 * 384 * v.div_ceil(4096);
    assert_eq!(partielles, 1_358_954_496);
    let m =
        majorant_memoire_gen([768, 768, 2, 2], 1, 2, 1, 384, 384, 0, false).expect("représentable");
    let borne = (216 + 48 + lignes) * v + partielles;
    assert!(m >= borne, "majorant {m} < lignes + partielles {borne}");
}

fn analyse_tube_dossier(nom: &str, nlignes: usize) -> PathBuf {
    let d = dossier_test(nom);
    std::fs::write(
        d.join("tube.json"),
        serde_json::to_vec(&json!({
            "cotes": [2, 2, 2, 2],
            "tube_obs": {"plan": [0, 1], "rmax": 1, "tmax": 1, "every": 1},
            "betas": [1.0],
            "nmeas": nlignes,
        }))
        .unwrap(),
    )
    .unwrap();
    let mut csv =
        String::from("V,plaq01,plaq02,plaq23,P2,absP2,P2sq,P3,absP3,P3sq,frac_vortex01,Wt_1x1\n");
    for i in 0..nlignes {
        let x = 0.01 * (i as f64 + 1.0);
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{}\n",
            1.0 + x,
            0.5 + x,
            0.4 + x,
            0.3 + x,
            0.1 + x,
            0.2 + x,
            0.3 + x,
            0.1 - x,
            0.2 + x,
            0.3 - x,
            0.25 + x,
            0.4 + x
        ));
    }
    std::fs::write(d.join("tube.csv"), csv).unwrap();
    d
}

fn lancer_analyse_tube(d: &Path) -> (i32, String, String) {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/tube/analyse_tube.py");
    let o = std::process::Command::new("python3")
        .arg("-I")
        .arg(&script)
        .arg(d)
        .output()
        .unwrap();
    (
        o.status.code().unwrap_or(RC_SIGNAL),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

#[test]
fn analyse_tube_refuse_moins_de_vingt_observations() {
    // 24 : moins de 20 observations ⟹ blocs vides ⟹ NaN/Infinity avec le code 0 ; refus explicite attendu
    if !python_numpy() {
        return;
    }
    let d = analyse_tube_dossier("analyse_courte", 2);
    let (rc, out, err) = lancer_analyse_tube(&d);
    assert_ne!(rc, 0, "analyse courte acceptée : {out} {err}");
    assert!(err.contains("20"), "message de refus sans « 20 » : {err}");
    assert!(
        !out.contains("NaN") && !out.contains("Infinity"),
        "ligne de résultat non finie : {out}"
    );
    // contrôle positif : 20 observations sont analysées, résultat fini, code 0
    let d = analyse_tube_dossier("analyse_vingt", 20);
    let (rc, out, err) = lancer_analyse_tube(&d);
    assert_eq!(rc, 0, "{err}");
    assert!(
        !out.contains("NaN") && !out.contains("Infinity"),
        "résultat non fini : {out}"
    );
}

#[test]
fn avertissement_entre_runs_exact() {
    // 15 : deux runs partagent un flux ssi (g′ − g)·7919⁻¹ mod 2⁶⁴ ∈ K − K′ ; jeux disjoints : g_j = g₀ + 7919·2²⁰·j
    use kvu::su2::tube::cli::taches_apparie;
    let cles = |g: u64, nb: usize, nr: usize| -> HashSet<u64> {
        taches_apparie(g, nb, nr)
            .unwrap()
            .into_iter()
            .map(|t| t.3)
            .collect()
    };
    let g = 1u64;
    // le contre-exemple de la relecture : deux β, une réplique, graines 1 et 1 + 7919·1000 : flux partagés
    assert!(!cles(g, 2, 1).is_disjoint(&cles(g + 7919 * 1000, 2, 1)));
    // la recommandation : décalage 7919·2²⁰ (K ⊂ [0, 2¹⁹) dans le domaine admis), même au plus grand lot admis
    for j in 1..4u64 {
        let g2 = g.wrapping_add(7919u64.wrapping_mul(1 << 20).wrapping_mul(j));
        assert!(
            cles(g, 256, 1024).is_disjoint(&cles(g2, 256, 1024)),
            "j = {j}"
        );
    }
    let note = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/v0.2/notes/tube.md"),
    )
    .unwrap();
    assert!(
        note.contains("7919⁻¹") && note.contains("7919·2²⁰·j"),
        "note sans la condition exacte"
    );
}

/// I_n(x) par la série (mêmes 60 termes que ti_etalonnage.py).
fn bessel_i(n: i32, x: f64) -> f64 {
    let mut s = 0.0;
    let mut fk = 1.0f64; // k!
    for k in 0..60 {
        if k > 0 {
            fk *= k as f64;
        }
        let mut fkn = 1.0f64; // (k + n)!
        for i in 2..=(k + n) {
            fkn *= i as f64;
        }
        s += (x / 2.0).powi(2 * k + n) / (fk * fkn);
    }
    s
}

/// Fixture INDÉPENDANTE de l'étalonnage (MINEURE 18) : intégrandes construits pour que la quadrature de Simpson du script
/// reproduise EXACTEMENT les références de couplage fort Δ_SC et log R_SC aux points pairs de la grille (h = 1/8, β ≤ 1) :
/// le script non muté rend P1, P1_abs et P2 VERTS ; le mutant (intégrande attribué au point voisin) doit les rendre ROUGES.
fn fixture_couplage_fort(d: &Path) -> Vec<PathBuf> {
    let h = 0.125f64;
    let b: Vec<f64> = (0..=8).map(|i| i as f64 * h).collect();
    let t = |x: f64| {
        if x == 0.0 {
            0.0
        } else {
            bessel_i(2, x) / bessel_i(1, x)
        }
    };
    // intégrande y tel que Simpson cumulé aux points pairs = F(b_k) ; y_0 = 0 (imposé par le script), y pair = F′, y impair déduit
    let integrande = |f: &dyn Fn(f64) -> f64| -> Vec<f64> {
        let fp = |x: f64| (f(x + 1e-6) - f(x - 1e-6)) / 2e-6;
        let mut y = vec![0.0; 9];
        for k in (2..=8).step_by(2) {
            y[k] = fp(b[k]);
        }
        for k in (1..=7).step_by(2) {
            y[k] = (3.0 * (f(b[k + 1]) - f(b[k - 1])) / h - y[k - 1] - y[k + 1]) / 4.0;
        }
        y
    };
    let delta = |x: f64| 384.0 * t(x).powi(4) + 1536.0 * t(x).powi(8);
    let y_delta = integrande(&delta); // y = −(16 V_G0 − V_G4)
    let cotes = [
        [2usize, 2, 2, 2],
        [4, 2, 2, 2],
        [4, 4, 2, 2],
        [4, 4, 4, 2],
        [4, 4, 4, 4],
    ];
    let mut dirs = Vec::new();
    for (j, c) in cotes.iter().enumerate() {
        let vol = (c[0] * c[1] * c[2] * c[3]) as f64;
        let a = (c[2] * c[3]) as i32; // plan ajouté 23
        let logr = move |x: f64| -2.0 * vol / a as f64 * t(x).powi(a);
        let z = integrande(&logr); // z = −DV
        let res: Vec<Value> = (1..=8)
            .map(|i| {
                let v0 = if j == 4 {
                    6.0 * vol + y_delta[i]
                } else {
                    6.0 * vol
                };
                json!({"beta": b[i], "V0": [v0, 1e-9], "V1": [v0 - z[i], 1e-9], "DV": [-z[i], 1e-9],
                       "par_replique": {"V0": [[v0, 1e-9]], "V1": [[v0 - z[i], 1e-9]]}})
            })
            .collect();
        let x = d.join(format!("sc{j}"));
        std::fs::create_dir_all(&x).unwrap();
        let jj = json!({"cotes": c, "secteur0": "aucune", "secteur1": "23", "resultats": res});
        std::fs::write(
            x.join("apparie.json"),
            serde_json::to_vec_pretty(&jj).unwrap(),
        )
        .unwrap();
        dirs.push(x);
    }
    dirs
}

#[test]
fn etalonnage_mutant_rouge_sur_fixture_independante() {
    if !python_numpy() {
        return;
    }
    let d = dossier_test("etalonnage_sc");
    let dirs = fixture_couplage_fort(&d);
    let juges = |j: &Value| -> Vec<(String, String)> {
        j["verdicts"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|v| {
                let n = v[0].as_str().unwrap();
                n == "P1" || n == "P1_abs" || n.starts_with("P2_")
            })
            .map(|v| {
                (
                    v[0].as_str().unwrap().to_string(),
                    v[2].as_str().unwrap().to_string(),
                )
            })
            .collect()
    };
    let (rc, out, err) = etalonnage(&d, &dirs, &[]);
    assert_eq!(rc, 0, "{err}");
    let j: Value = serde_json::from_str(&out).unwrap();
    let v = juges(&j);
    assert!(v.len() >= 8, "{v:?}");
    assert!(
        v.iter().all(|x| x.1 == "VERT"),
        "fixture non conforme au couplage fort : {v:?}"
    );
    let (rc, out, err) = etalonnage(&d, &dirs, &["--mutant-decalage"]);
    assert_eq!(rc, 0, "{err}");
    let j: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(
        j["mutant_rouge"],
        true,
        "mutant non rejeté : {:?}",
        juges(&j)
    );
}
