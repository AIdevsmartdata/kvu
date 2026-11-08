//! R6 : mesures imbriquées, conservation physique et migration du JSON.
use kvu::commun::{BackendKind, Depart, Erreur, Graine, Precision, Resultat};
use kvu::gpu::{Appareil, OptionsCompil};
use kvu::su2::run::{self, ConfigRun, MesureEnLigne, ProfilRun};
use kvu::su2::{ConfigLot, ParamChaine};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn rep(nom: &str) -> PathBuf {
    let p = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("banc_r6_{nom}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

fn config(outdir: PathBuf, backend: BackendKind) -> ConfigRun {
    ConfigRun {
        lot: ConfigLot { l: 4, chaines: (0..2).map(|i| ParamChaine {
            beta: 2.5, graine: Graine(28_102_600 + i), depart: Depart::Chaud,
        }).collect(), nor: 3, precision: Precision::F64,
            options: OptionsCompil::compat_production(backend), mutant_metro: false },
        therm: 4, nmeas: 5, skip: 2, dump_every: 1, dump_initial: true,
        format_dump: "c16".into(), zstd: false, reprise_every: 4, outdir, bord: None,
    }
}

fn physiques(base: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn visiter(base: &Path, p: &Path, out: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for e in fs::read_dir(p).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() { visiter(base, &p, out); }
            else if matches!(p.extension().and_then(|x| x.to_str()), Some("npy" | "csv" | "champs"))
                || p.file_name().and_then(|x| x.to_str()) == Some("reprise.json") {
                out.insert(p.strip_prefix(base).unwrap().to_path_buf(), fs::read(p).unwrap());
            }
        }
    }
    let mut r = BTreeMap::new();visiter(base, base, &mut r);r
}

fn verifier_complete(base: &Path) {
    // COMPLETE référence le manifeste, dont les temps et le chemin diffèrent.
    // Son intégrité est exigée séparément des octets physiques.
    let sha = kvu::heritage::manifest::sha256_fichier(&base.join("manifest.json")).unwrap();
    assert_eq!(fs::read_to_string(base.join("COMPLETE")).unwrap(), format!("rc=0\nmanifest_sha256={sha}\n"));
}

fn identiques(a: &BTreeMap<PathBuf, Vec<u8>>, b: &BTreeMap<PathBuf, Vec<u8>>) {
    assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
    for (nom, octets) in a { assert!(octets == &b[nom], "données physiques différentes : {}", nom.display()); }
}

#[test]
fn compteurs_et_durees_refusent_les_poisons_sans_clamp() {
    let p = ProfilRun { duree_totale_ns: 1000, duree_pas_ns: 600, pas_executes: 10,
        hb_executes: 10, or_executes: 30, chaines: 2, nor: 3, termine: true };
    p.valider(10, 2, 3, true).unwrap();
    assert!((p.secondes_pas() / p.secondes_totales() - 0.6).abs() < 1e-15);
    let mut poisons = vec![];
    let mut q = p.clone();q.duree_pas_ns = 1001;poisons.push(q);
    let mut q = p.clone();q.duree_pas_ns = 0;poisons.push(q);
    let mut q = p.clone();q.duree_totale_ns = 0;poisons.push(q);
    let mut q = p.clone();q.pas_executes = 9;poisons.push(q);
    let mut q = p.clone();q.hb_executes = 9;poisons.push(q);
    let mut q = p.clone();q.or_executes = 29;poisons.push(q);
    let mut q = p.clone();q.chaines = 1;poisons.push(q);
    let mut q = p.clone();q.nor = 4;poisons.push(q);
    let mut q = p.clone();q.termine = false;poisons.push(q);
    for poison in poisons {
        let avant = poison.clone();
        assert!(poison.valider(10, 2, 3, true).is_err());
        assert_eq!(poison, avant, "le validateur ne corrige pas son entrée");
    }
    assert!(p.valider(0, 2, 3, true).is_err());
    assert!(p.valider(u64::MAX, 2, 3, true).is_err());
    for valeur in ["NaN", "Infinity", "-1", "0.5", "18446744073709551616"] {
        let texte = serde_json::to_string(&p).unwrap().replace("\"duree_pas_ns\":600", &format!("\"duree_pas_ns\":{valeur}"));
        assert!(serde_json::from_str::<ProfilRun>(&texte).is_err(), "poison {valeur}");
    }
}

#[test]
fn profil_cpu_conserve_champs_csv_reprise_et_compteurs() {
    let d = rep("physique");let normal = config(d.join("normal"), BackendKind::Cpu);
    let profile = ConfigRun { outdir: d.join("profile"), ..normal.clone() };
    let a = run::executer(None, &normal, &mut [], false).unwrap();
    let (b, fini, p) = run::executer_profile(None, &profile, &mut [], None, None).unwrap();
    assert!(fini);p.valider(14, 2, 3, true).unwrap();
    assert_eq!((a.pas, a.mesures, a.dumps, a.plafond), (b.pas, b.mesures, b.dumps, b.plafond));
    let pa = physiques(&normal.outdir);let pb = physiques(&profile.outdir);
    assert!(pa.contains_key(Path::new("reprise.json")) && pa.len() > 10);
    identiques(&pa, &pb);verifier_complete(&normal.outdir);verifier_complete(&profile.outdir);
    println!("PROFIL_CPU {}", serde_json::to_string(&p).unwrap());
    let _ = fs::remove_dir_all(d);
}

#[test]
fn interruption_profilee_conserve_le_checkpoint_sans_complete() {
    let d = rep("interruption");let normal = config(d.join("normal"), BackendKind::Cpu);
    let profile = ConfigRun { outdir: d.join("profile"), ..normal.clone() };
    let (a, fini_a) = run::executer_jusqua(None, &normal, &mut [], false, Some(8)).unwrap();
    let (b, fini_b, p) = run::executer_profile(None, &profile, &mut [], Some(8), None).unwrap();
    assert!(!fini_a && !fini_b);assert_eq!(a.pas, b.pas);p.valider(a.pas, 2, 3, false).unwrap();
    assert!(!normal.outdir.join("COMPLETE").exists() && !profile.outdir.join("COMPLETE").exists());
    identiques(&physiques(&normal.outdir), &physiques(&profile.outdir));
    let _ = fs::remove_dir_all(d);
}

struct PoisonMesure;
impl MesureEnLigne for PoisonMesure {
    fn nom(&self) -> &str { "poison_r6" }
    fn periode(&self) -> usize { 1 }
    fn mesurer(&mut self, _: Option<&Appareil>, _: Option<&kvu::su2::ChampGpu>, _: Option<&kvu::commun::ChampHote>,
        _: usize, _: u64, _: &Path) -> Resultat<()> { Err(Erreur::Config("poison mesure R6".into())) }
}

#[test]
fn erreur_profilee_ne_produit_ni_complete_ni_profil_valide() {
    let d = rep("erreur");let normal = config(d.join("normal"), BackendKind::Cpu);
    let profile = ConfigRun { outdir: d.join("profile"), ..normal.clone() };
    let a = run::executer(None, &normal, &mut [Box::new(PoisonMesure)], false).unwrap_err();
    let b = run::executer_profile(None, &profile, &mut [Box::new(PoisonMesure)], None, None).unwrap_err();
    assert!(a.to_string().contains("poison mesure R6") && b.to_string().contains("poison mesure R6"));
    assert!(!normal.outdir.join("COMPLETE").exists() && !profile.outdir.join("COMPLETE").exists());
    let vide = ConfigRun { therm: 0, nmeas: 0, outdir: d.join("vide"), ..normal };
    assert!(run::executer_profile(None, &vide, &mut [], None, None).is_err());
    assert!(!vide.outdir.exists());
    let _ = fs::remove_dir_all(d);
}

fn document_cli(sans_mur: bool) -> Value {
    let mut c = Command::new(env!("CARGO_BIN_EXE_kvu"));
    c.args(["bench", "su2", "--backend", "cpu", "--L", "4", "--precision", "f64", "--chaines", "1", "--budget-s", "60", "--cible-s", "1", "--ref-bin", "aucune", "--json"]);
    if sans_mur { c.arg("--sans-mur"); }
    let o = c.output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let txt = String::from_utf8(o.stdout).unwrap();
    let debut = txt.find('{').expect("JSON présent");
    let v: Value = serde_json::from_str(&txt[debut..]).unwrap();
    println!("JSON_BANC_R6 {}", v);
    v
}

#[test]
fn json_versionne_publie_durees_brutes_et_sans_mur_reste_micro() {
    for sans_mur in [false, true] {
        let d = document_cli(sans_mur);assert_eq!(d["schema"], "kvu-banc/2");
        let m = &d["mesures"][0];assert_eq!(m["schema_mesure"], "kvu-banc-su2/2");
        let micro = &m["microbench"];
        let s = micro["hb_s_par_sweep"].as_f64().unwrap() + 4.0 * micro["or_s_par_sweep"].as_f64().unwrap();
        assert!((micro["pas_par_s"].as_f64().unwrap() * s - 1.0).abs() < 1e-12);
        if sans_mur {
            assert_eq!(m["mode_mesure"], "micro_seul");assert!(m.get("mur").is_none() && m.get("profilage").is_none());
            assert_eq!(m["pas_par_s"], micro["pas_par_s"]);
        } else {
            assert_eq!(m["mode_mesure"], "apparie");
            let p: ProfilRun = serde_json::from_value(m["profilage"].clone()).unwrap();
            p.valider(p.pas_executes, 1, 4, true).unwrap();assert!(p.pas_executes >= 40);
            let n = p.pas_executes as f64;
            assert!((m["pas_par_s"].as_f64().unwrap() * p.secondes_pas() / n - 1.0).abs() < 1e-12);
            assert!((m["mur"]["pas_par_s"].as_f64().unwrap() * p.secondes_totales() / n - 1.0).abs() < 1e-12);
            assert!((m["rapport_mur_sur_noyaux"].as_f64().unwrap() - p.secondes_pas()/p.secondes_totales()).abs() < 1e-12);
        }
    }
}

#[test]
fn gate_reference_native_reste_liee_au_microbench() {
    use std::os::unix::fs::PermissionsExt;
    let d = rep("reference");let reference = d.join("reference_synthetique.sh");
    fs::write(&reference, "#!/bin/sh\nprintf 'BENCH L=4 hb_s_par_sweep=0.01 or_s_par_sweep=0.004\\n'\n").unwrap();
    fs::set_permissions(&reference, fs::Permissions::from_mode(0o700)).unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_kvu"))
        .args(["bench", "su2", "--backend", "cpu", "--L", "4", "--precision", "f64", "--chaines", "1", "--budget-s", "60", "--cible-s", "1", "--ref-bin"])
        .arg(&reference).arg("--json").output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let txt = String::from_utf8(o.stdout).unwrap();
    let doc: Value = serde_json::from_str(&txt[txt.find('{').unwrap()..]).unwrap();
    let m = &doc["mesures"][0];
    let attendu = 0.026 / m["microbench"]["s_par_pas"].as_f64().unwrap();
    let observe = m["ratio_pas_unifie_sur_reference"].as_f64().unwrap();
    assert!((observe / attendu - 1.0).abs() < 1e-12);
    assert_eq!(m["gate_non_regression"], attendu >= 0.9);
    println!("GATE_MICRO_REFERENCE_SYNTHETIQUE {}", m);
    let _ = fs::remove_dir_all(d);
}

#[test]
fn rapports_v2_utilisent_microbench_pour_reference_et_gains() {
    let d = rep("rapports");fs::create_dir(d.join("acemagic")).unwrap();
    let mesure = |p: &str, micro: f64, apparie: f64, ratio: Option<f64>| {
        let mut m = json!({"schema_mesure":"kvu-banc-su2/2","mode_mesure":"apparie","famille":"su2","L":4,"precision":p,"chaines":1,
            "liens_par_s_hb":1.0,"liens_par_s_or":1.0,"pas_par_s":apparie,"pas_chaines_par_s":apparie,"memoire_champ_mo":1.0,
            "microbench":{"pas_par_s":micro,"pas_chaines_par_s":micro},
            "mur":{"pas_par_s":apparie/2.0},"rapport_mur_sur_noyaux":0.5,"profilage":{"duree_totale_ns":1000,"duree_pas_ns":500,"pas_executes":40}});
        if let Some(r) = ratio { m["reference"]=json!({"binaire":"/ref","pas_par_s":100.0});m["ratio_pas_unifie_sur_reference"]=json!(r);m["gate_non_regression"]=json!(r>=0.9); }
        m
    };
    let doc = json!({"schema":"kvu-banc/2","banc":"su2","machine":"acemagic","backend":"cpu","arch":"cpu","budget_s":60.0,"duree_s":1.0,"fenetres":1,
        "mesures":[mesure("f64",95.0,2.0,Some(0.95)),mesure("f32",190.0,2000.0,None)]});
    fs::write(d.join("acemagic/su2_acemagic.json"), doc.to_string()).unwrap();
    let base=Path::new(env!("CARGO_MANIFEST_DIR")).join("validation");
    for (nom, script) in [("banc","banc/rapport_banc.py"),("integration","integration/rapport_bench.py")] {
        let sortie=d.join(format!("{nom}.md"));
        let o=Command::new("python3").arg(base.join(script)).arg(&d).arg("--sortie").arg(&sortie).output().unwrap();
        assert!(o.status.success(),"{}",String::from_utf8_lossy(&o.stderr));
        let txt=fs::read_to_string(sortie).unwrap();assert!(txt.contains("0.950") && txt.contains("PASS") && txt.contains("×2.00"),"{txt}");
        assert!(!txt.contains("×1000.00"),"gain apparié utilisé à tort : {txt}");
    }
    for (cle, erreur) in [("microbench", "sans microbench"), ("schema_mesure", "non versionnée"), ("mode_mesure", "mode de mesure")] {
        let mut poison=doc.clone();poison["mesures"][0].as_object_mut().unwrap().remove(cle);
        fs::write(d.join("acemagic/su2_acemagic.json"),poison.to_string()).unwrap();
        for script in ["banc/rapport_banc.py","integration/rapport_bench.py"] {
            let o=Command::new("python3").arg(base.join(script)).arg(&d).arg("--sortie").arg(d.join("poison.md")).output().unwrap();
            assert!(!o.status.success());assert!(String::from_utf8_lossy(&o.stderr).contains(erreur));
        }
    }
    let mut ancien=doc;ancien.as_object_mut().unwrap().remove("schema");
    for m in ancien["mesures"].as_array_mut().unwrap() {
        let micro=m["microbench"].clone();
        m["pas_par_s"]=micro["pas_par_s"].clone();m["pas_chaines_par_s"]=micro["pas_chaines_par_s"].clone();
        for cle in ["microbench","schema_mesure","mode_mesure","profilage"] { m.as_object_mut().unwrap().remove(cle); }
    }
    fs::write(d.join("acemagic/su2_acemagic.json"),ancien.to_string()).unwrap();
    for script in ["banc/rapport_banc.py","integration/rapport_bench.py"] {
        let sortie=d.join("ancien.md");
        let o=Command::new("python3").arg(base.join(script)).arg(&d).arg("--sortie").arg(&sortie).output().unwrap();
        assert!(o.status.success(),"{}",String::from_utf8_lossy(&o.stderr));
        assert!(fs::read_to_string(sortie).unwrap().contains("×2.00"));
    }
    let _=fs::remove_dir_all(d);
}

#[test]
#[ignore = "GPU : garde mémoire et verrou coordonnés par le pilote"]
fn profil_gpu_conserve_les_sorties_au_bit() {
    let backend: BackendKind=std::env::var("KVU_TEST_BACKEND").expect("backend explicite").parse().unwrap();
    assert!(matches!(backend,BackendKind::Hip|BackendKind::Cuda));
    let app=kvu::gpu::ouvrir(backend,0).unwrap();
    let chemin=if backend==BackendKind::Hip {"/tmp/kevinotron_gpu.lock"} else {"/tmp/gamer_gpu.lock"};
    let garde=kvu::machine::VerrouGpu::nouveau(Path::new(chemin)).acquerir(std::time::Duration::from_secs(600)).unwrap();
    let d=rep("gpu");let normal=config(d.join("normal"),backend);let profile=ConfigRun {outdir:d.join("profile"),..normal.clone()};
    run::executer_sous_verrou(Some(&app),&normal,&mut [],&garde).unwrap();
    let (_,fini,p)=run::executer_profile(Some(&app),&profile,&mut [],None,Some(&garde)).unwrap();
    assert!(fini);p.valider(14,2,3,true).unwrap();identiques(&physiques(&normal.outdir),&physiques(&profile.outdir));
    verifier_complete(&normal.outdir);verifier_complete(&profile.outdir);
    println!("PROFIL_GPU {}",serde_json::to_string(&p).unwrap());let _=fs::remove_dir_all(d);
}
