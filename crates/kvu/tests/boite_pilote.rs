//! Tests du pilote Python : faux kvu uniquement, aucun GPU ni accès réseau.

use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

struct Temp(PathBuf);

impl Temp {
    fn nouveau(nom: &str) -> Self {
        static COMPTEUR: AtomicUsize = AtomicUsize::new(0);
        let racine = Path::new(env!("CARGO_TARGET_TMPDIR"));
        let chemin = racine.join(format!(
            "boite_pilote_{nom}_{}_{}_{}",
            std::process::id(),
            COMPTEUR.fetch_add(1, Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&chemin).unwrap();
        Self(chemin)
    }

    fn joindre(&self, nom: &str) -> PathBuf {
        self.0.join(nom)
    }

    fn csv(&self, texte: &str) -> PathBuf {
        let fichier = self.joindre("taches.csv");
        fs::write(&fichier, texte).unwrap();
        fichier
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        // Nettoyage limité aux fichiers temporaires propres à ce test.
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn racine() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn python_disponible() -> bool {
    match Command::new("python3").args(["-I", "--version"]).output() {
        Ok(sortie) if sortie.status.success() => true,
        _ => {
            eprintln!("SKIP : python3 est absent ou inutilisable");
            false
        }
    }
}

fn pilote(taches: &Path, sortie: &Path, cibles: &[(&str, &Path)]) -> Command {
    let mut cmd = Command::new("python3");
    cmd.current_dir(racine())
        .args(["-I", "scripts/pilote_boites.py"])
        .arg(taches)
        .arg("--sortie")
        .arg(sortie);
    for (nom, binaire) in cibles {
        cmd.arg("--cible")
            .arg(format!("{nom}=local:{}", binaire.display()));
    }
    // L'environnement de chaque enfant est isolé de celui des autres tests.
    cmd.env_remove("FAUX_ECHEC_GRAINE")
        .env("FAUX_SOMMEIL", "0")
        .env_remove("FAUX_SANS_COMPLETE")
        .env_remove("FAUX_PID")
        .env_remove("FAUX_MANIFEST_INVALIDE");
    cmd
}

fn verifier_code(sortie: &Output, code: i32) {
    assert_eq!(
        sortie.status.code(),
        Some(code),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&sortie.stdout),
        String::from_utf8_lossy(&sortie.stderr)
    );
}

fn index(sortie: &Path) -> Value {
    serde_json::from_slice(&fs::read(sortie.join("index.json")).unwrap()).unwrap()
}

fn lignes_plan(sortie: &Output) -> Vec<Vec<String>> {
    String::from_utf8(sortie.stdout.clone())
        .unwrap()
        .lines()
        .filter(|s| s.starts_with('L'))
        .map(|s| s.split_whitespace().map(str::to_owned).collect())
        .collect()
}

#[cfg(unix)]
fn faux_kvu(t: &Temp, nom: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let chemin = t.joindre(nom);
    fs::write(
        &chemin,
        r#"#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
import time

args = sys.argv[3:]
p = dict(zip(args[::2], args[1::2]))
assert sys.argv[1:3] == ['su2', 'boite']
assert '--bords' not in sys.argv
assert p['--backend'] == 'auto'
assert all(k in p for k in ('--therm', '--nmeas', '--skip', '--nor', '--options', '--profil-tous'))
chaine = p['--chaines'].split(',')
sortie = Path(p['--outdir'])
sortie.mkdir()
debut = time.time()
if os.environ.get('FAUX_PID'):
    # écriture ATOMIQUE : le test ne doit jamais lire un fichier créé mais encore vide (Path.write_text crée puis écrit)
    temporaire = Path(os.environ['FAUX_PID'] + '.tmp')
    temporaire.write_text(str(os.getpid()))
    os.replace(temporaire, os.environ['FAUX_PID'])
print('FAUX nom=' + Path(sys.argv[0]).name + ' lot=' + sortie.name, flush=True)
time.sleep(float(os.environ.get('FAUX_SOMMEIL', '0')))
rc = 5 if any(c.split(':')[1] == os.environ.get('FAUX_ECHEC_GRAINE') for c in chaine) else 0
for i in range(len(chaine)):
    dossier = sortie / ('c%03d' % i)
    dossier.mkdir()
    (dossier / 'mesures.csv').write_text('pas,pc,w2\n1,0.6,0.4\n')
    (dossier / 'profil.csv').write_text('distance,pc\n1,0.6\n')
manifest = dict(verrou='pris_ici', debut_unix=debut, fin_unix=time.time(), statut='ok' if rc == 0 else 'echec', rc=rc)
(sortie / 'manifest.json').write_text('{illisible' if os.environ.get('FAUX_MANIFEST_INVALIDE') else json.dumps(manifest))
(sortie / 'manifest_regen.json').write_text('{}')
if rc == 0 and not os.environ.get('FAUX_SANS_COMPLETE'):
    (sortie / 'COMPLETE').write_text('')
print('PERF boite ell=%s beta=%s K=%d backend=cpu precision=f64 pas=31000 s_boucle=1.0 ms_par_pas_et_chaine=0.0124 liens_libres_par_s=1.2e8' % (p['--ell'], p['--beta'], len(chaine)), flush=True)
sys.exit(rc)
"#,
    )
    .unwrap();
    fs::set_permissions(&chemin, fs::Permissions::from_mode(0o755)).unwrap();
    chemin
}

#[test]
fn plan_deterministe_sans_ecriture_et_temps_a_la_main() {
    if !python_disponible() {
        return;
    }
    let t = Temp::nouveau("plan");
    let mut csv = String::from("# commentaires et lignes vides\n\nell,beta,bord,graine\n");
    for (ell, beta) in [(2, "2.5"), (4, "2.6"), (6, "2.7")] {
        for graine in 0..4 {
            csv.push_str(&format!("{ell},{beta},cold,{graine}\n"));
        }
    }
    let fichier = t.csv(&csv);
    let sortie = t.joindre("sortie_absente");
    let absent = t.joindre("ne_pas_executer");
    let lancer = || {
        pilote(&fichier, &sortie, &[("lent", &absent), ("vite", &absent)])
            .args([
                "--dry-run",
                "--chaines-max",
                "3",
                "--fenetre-s",
                "600",
                "--debit",
                "lent=5e7",
                "--debit",
                "vite=1e8",
            ])
            .output()
            .unwrap()
    };
    let avant: HashSet<_> = fs::read_dir(&t.0)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    let a = lancer();
    let b = lancer();
    verifier_code(&a, 0);
    verifier_code(&b, 0);
    assert_eq!(a.stdout, b.stdout);
    assert!(String::from_utf8_lossy(&a.stdout).ends_with("PLAN OK\n"));
    assert!(!sortie.exists());
    let apres: HashSet<_> = fs::read_dir(&t.0)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(avant, apres);
    let lignes = lignes_plan(&a);
    assert_eq!(lignes.len(), 6);
    assert_eq!(
        lignes
            .iter()
            .map(|l| l[2].parse::<usize>().unwrap())
            .sum::<usize>(),
        12
    );
    for ligne in &lignes {
        assert_eq!(ligne.len(), 6);
        assert_eq!(ligne[2], "2", "lots équilibrés : 2+2 et non 3+1");
        assert!(ligne[5].parse::<f64>().unwrap() <= 600.0);
    }
    // Vérification indépendante de l'affectation LPT avec les deux débits.
    let mut charges = [0.0_f64, 0.0_f64];
    for ligne in lignes {
        let ell = ligne[3].parse::<f64>().unwrap();
        let travail = 31000.0 * 2.0 * 4.0 * ell * (ell - 1.0).powi(3) * 3.0;
        let durees = [30.0 + travail / 5e7, 30.0 + travail / 1e8];
        let choix = if charges[0] + durees[0] <= charges[1] + durees[1] {
            0
        } else {
            1
        };
        assert_eq!(ligne[1], ["lent", "vite"][choix]);
        assert!((ligne[5].parse::<f64>().unwrap() - durees[choix]).abs() < 1e-7);
        charges[choix] += durees[choix];
    }
    // Valeur à la main : 4*8*7³ = 10976 liens ;
    // 31000*32*10976*3 = 32664576000, /50000000 + 30 = 683.29152 s.
    let fichier = t.csv("ell,beta,bord,graine\n8,2.5,cold,0\n");
    let mut contenu = fs::read_to_string(&fichier).unwrap();
    for graine in 1..32 {
        contenu.push_str(&format!("8,2.5,cold,{graine}\n"));
    }
    fs::write(&fichier, contenu).unwrap();
    let chiffre = pilote(&fichier, &sortie, &[("test", &absent)])
        .args(["--dry-run", "--fenetre-s", "1000"])
        .output()
        .unwrap();
    verifier_code(&chiffre, 0);
    let lignes = lignes_plan(&chiffre);
    assert_eq!(lignes.len(), 1);
    assert_eq!(lignes[0][2], "32");
    let estime = lignes[0][5].parse::<f64>().unwrap();
    assert!((estime - 683.29152).abs() < 1e-8, "{estime}");
    assert!(!sortie.exists());
}

#[test]
fn refus_usage_csv_fenetre_et_memoire_avant_tout_lancement() {
    if !python_disponible() {
        return;
    }
    let t = Temp::nouveau("refus");
    let absent = t.joindre("interdit");
    let sortie = t.joindre("sortie");
    for (csv, mot, numero) in [
        (
            "ell,beta,bord,graine,nmeas\n8,2.5,cold,0,999999999999\n",
            "fenêtre",
            None,
        ),
        (
            "ell,beta,bord,graine\n# commentaire\n\n3,2.5,cold,0\n",
            "pair",
            Some(4),
        ),
        ("ell,beta,bord,graine\n2,2.5,inconnu,0\n", "bord", Some(2)),
        (
            "ell,beta,bord,graine\n2,2.5,cold,0\n2,2.50,cold,0\n",
            "doublon",
            Some(3),
        ),
        ("ell,beta,graine\n2,2.5,0\n", "en-tête", Some(1)),
        ("ell,beta,bord,graine\n2,nan,cold,0\n", "positif", Some(2)),
        ("ell,beta,bord,graine\n2,inf,cold,0\n", "positif", Some(2)),
        ("ell,beta,bord,graine\n2,2.5,cold,-1\n", "graine", Some(2)),
        (
            "ell,beta,bord,graine,nmeas\n2,2.5,cold,0,0\n",
            "nmeas",
            Some(2),
        ),
        (
            "ell,beta,bord,graine,options\n2,2.5,cold,0,autre\n",
            "options",
            Some(2),
        ),
        (
            "ell,beta,bord,graine\n2,2.5,cold,0,extra\n",
            "colonnes",
            Some(2),
        ),
    ] {
        let fichier = t.csv(csv);
        let resultat = pilote(&fichier, &sortie, &[("test", &absent)])
            .output()
            .unwrap();
        verifier_code(&resultat, 2);
        let erreur = String::from_utf8_lossy(&resultat.stderr);
        assert!(erreur.contains(mot), "{erreur}");
        if let Some(n) = numero {
            assert!(erreur.contains(&format!("ligne {n}")), "{erreur}");
        }
        assert!(!sortie.exists());
    }
    let fichier = t.csv("ell,beta,bord,graine\n2,2.5,cold,0\n");
    let resultat = pilote(&fichier, &sortie, &[]).output().unwrap();
    verifier_code(&resultat, 2);
    assert!(String::from_utf8_lossy(&resultat.stderr).contains("cible"));
    let memoire = pilote(&fichier, &sortie, &[("test", &absent)])
        .args(["--mem-gpu-go", "0.000001"])
        .output()
        .unwrap();
    verifier_code(&memoire, 2);
    let erreur = String::from_utf8_lossy(&memoire.stderr);
    assert!(
        erreur.contains("mémoire")
            && erreur.contains("ell=2")
            && erreur.contains("beta=2.5")
            && erreur.contains("temps estimé")
    );
    assert!(!sortie.exists());
    let aide = Command::new("python3")
        .current_dir(racine())
        .args(["-I", "scripts/pilote_boites.py", "--aide"])
        .output()
        .unwrap();
    verifier_code(&aide, 0);
    assert!(String::from_utf8_lossy(&aide.stdout).contains("Mode de test"));
}

#[test]
#[cfg(unix)]
fn deux_cibles_execution_chevauchement_index_et_reprise() {
    if !python_disponible() {
        return;
    }
    let t = Temp::nouveau("execution");
    // Espaces et apostrophe : aucun interpréteur de commandes en local.
    let a = faux_kvu(&t, "faux a' kvu");
    let b = faux_kvu(&t, "faux b kvu");
    let fichier = t
        .csv("ell,beta,bord,graine\n2,2.1,cold,11\n2,2.2,gauge,12\n2,2.3,flux,13\n2,2.4,haar,14\n");
    let sortie = t.joindre("sortie avec espaces");
    let depart = Instant::now();
    let resultat = pilote(&fichier, &sortie, &[("a", &a), ("b", &b)])
        .env("FAUX_SOMMEIL", "1")
        .output()
        .unwrap();
    let duree = depart.elapsed().as_secs_f64();
    verifier_code(&resultat, 0);
    let doc = index(&sortie);
    assert_eq!(doc["schema"], "pilote-boites/1");
    assert!(doc["session"].as_str().unwrap().starts_with("s_"));
    let lots = doc["lots"].as_array().unwrap();
    assert_eq!(lots.len(), 4);
    assert_eq!(doc["taches"].as_array().unwrap().len(), 4);
    assert!(doc["cibles"]["a"]["lots"].as_u64().unwrap() >= 1);
    assert!(doc["cibles"]["b"]["lots"].as_u64().unwrap() >= 1);
    let mut somme = 0.0;
    let mut journaux = Vec::new();
    for lot in lots {
        assert_eq!(lot["statut"], "ok");
        assert_eq!(lot["rc"], 0);
        assert_eq!(lot["verrou"], "pris_ici");
        assert_eq!(lot["perf"]["backend"], "cpu");
        assert_eq!(lot["perf"]["K"], "1");
        let dossier = sortie.join(lot["sortie"].as_str().unwrap());
        assert!(dossier.join("COMPLETE").exists());
        assert_eq!(
            fs::read_to_string(dossier.join("c000/mesures.csv")).unwrap(),
            "pas,pc,w2\n1,0.6,0.4\n"
        );
        assert!(dossier.join("c000/profil.csv").exists());
        assert!(dossier.join("manifest_regen.json").exists());
        let log = sortie
            .join("lots")
            .join(format!("{}.log", lot["id"].as_str().unwrap()));
        let texte = fs::read_to_string(&log).unwrap();
        let nom = if lot["cible"] == "a" {
            "faux a' kvu"
        } else {
            "faux b kvu"
        };
        assert!(texte.contains(nom), "{texte}");
        journaux.push((log, texte));
        let cmd = lot["commande"].as_array().unwrap();
        assert_eq!(cmd[1], "su2");
        assert_eq!(cmd[2], "boite");
        assert!(cmd.iter().any(|v| v == "--chaines"));
        assert!(!cmd.iter().any(|v| v == "--bords"));
        assert!(!cmd.iter().any(|v| v == "flock"));
        let debut = lot["debut_unix"].as_f64().unwrap();
        let fin = lot["fin_unix"].as_f64().unwrap();
        assert!(fin - debut >= 0.99);
        somme += lot["duree_s"].as_f64().unwrap();
    }
    assert!(
        duree < somme,
        "durée totale={duree}, somme des durées={somme}"
    );
    assert!(lots
        .iter()
        .enumerate()
        .any(|(i, x)| lots.iter().skip(i + 1).any(|y| {
            x["cible"] != y["cible"]
                && x["debut_unix"].as_f64().unwrap() < y["fin_unix"].as_f64().unwrap()
                && y["debut_unix"].as_f64().unwrap() < x["fin_unix"].as_f64().unwrap()
        })));
    for tache in doc["taches"].as_array().unwrap() {
        let lot = lots.iter().find(|l| l["id"] == tache["lot"]).unwrap();
        assert_eq!(tache["chaine"], 0);
        assert_eq!(lot["ell"], tache["ell"]);
        assert_eq!(lot["beta"], tache["beta"]);
        let args = lot["commande"].as_array().unwrap();
        let position = args.iter().position(|v| v == "--chaines").unwrap();
        assert_eq!(
            args[position + 1],
            format!("{}:{}", tache["bord"].as_str().unwrap(), tache["graine"])
        );
    }
    // Remplacer les faux par des exécutables qui laisseraient une preuve en cas de relance.
    let preuve = t.joindre("relance_interdite");
    let code = format!("#!/usr/bin/env python3\nfrom pathlib import Path\nPath({:?}).write_text('relancé')\nraise SystemExit(99)\n", preuve.to_str().unwrap());
    fs::write(&a, &code).unwrap();
    fs::write(&b, &code).unwrap();
    let reprise = pilote(&fichier, &sortie, &[("a", &a), ("b", &b)])
        .arg("--reprendre")
        .output()
        .unwrap();
    verifier_code(&reprise, 0);
    assert!(!preuve.exists());
    let doc_reprise = index(&sortie);
    assert_eq!(doc["session"], doc_reprise["session"]);
    for lot in doc_reprise["lots"].as_array().unwrap() {
        assert_eq!(lot["statut"], "deja_fait");
    }
    for (fichier, texte) in journaux {
        assert_eq!(fs::read_to_string(fichier).unwrap(), texte);
    }
    let sans = pilote(&fichier, &sortie, &[("a", &a), ("b", &b)])
        .output()
        .unwrap();
    verifier_code(&sans, 2);
    assert!(String::from_utf8_lossy(&sans.stderr).contains("--reprendre"));
}

#[test]
#[cfg(unix)]
fn echec_narrete_pas_les_autres_lots() {
    if !python_disponible() {
        return;
    }
    let t = Temp::nouveau("echec");
    let a = faux_kvu(&t, "faux_a");
    let b = faux_kvu(&t, "faux_b");
    let fichier =
        t.csv("ell,beta,bord,graine\n2,2.1,cold,10\n2,2.2,cold,11\n2,2.3,cold,12\n2,2.4,cold,13\n");
    let sortie = t.joindre("sortie");
    let resultat = pilote(&fichier, &sortie, &[("a", &a), ("b", &b)])
        .env("FAUX_ECHEC_GRAINE", "11")
        .output()
        .unwrap();
    verifier_code(&resultat, 1);
    let doc = index(&sortie);
    let lots = doc["lots"].as_array().unwrap();
    assert_eq!(lots.iter().filter(|l| l["statut"] == "ok").count(), 3);
    let echec = lots.iter().find(|l| l["statut"] == "echec").unwrap();
    assert_eq!(echec["rc"], 5);
    assert!(!sortie
        .join(echec["sortie"].as_str().unwrap())
        .join("COMPLETE")
        .exists());
    assert_eq!(echec["perf"]["backend"], "cpu");
    let journal = fs::read_to_string(sortie.join("pilote.log")).unwrap();
    assert_eq!(journal.matches("démarrage ").count(), 4);
    assert_eq!(journal.matches("échec ").count(), 1);
    assert_eq!(journal.matches(" fin ").count(), 3);
}

#[test]
#[cfg(unix)]
fn incomplets_conserves_suffixes_et_reprise_du_suffixe_termine() {
    if !python_disponible() {
        return;
    }
    let t = Temp::nouveau("incomplets");
    let faux = faux_kvu(&t, "faux");
    let fichier = t.csv("ell,beta,bord,graine\n2,2.5,cold,1\n");
    let sortie = t.joindre("sortie");
    let resultat = pilote(&fichier, &sortie, &[("a", &faux)])
        .env("FAUX_SANS_COMPLETE", "1")
        .output()
        .unwrap();
    verifier_code(&resultat, 1);
    let doc = index(&sortie);
    assert_eq!(doc["lots"][0]["statut"], "echec");
    assert_eq!(doc["lots"][0]["rc"], 0, "rc 0 seul ne suffit pas");
    let base = doc["lots"][0]["id"].as_str().unwrap().to_owned();
    let ancien_log = sortie.join("lots").join(format!("{base}.log"));
    let contenu_log = fs::read(&ancien_log).unwrap();
    let ancien_manifest = sortie.join("lots").join(&base).join("manifest.json");
    let contenu_manifest = fs::read(&ancien_manifest).unwrap();
    let incomplet = pilote(&fichier, &sortie, &[("a", &faux)])
        .arg("--reprendre")
        .output()
        .unwrap();
    verifier_code(&incomplet, 1);
    assert_eq!(index(&sortie)["lots"][0]["statut"], "incomplet_existant");
    // Une tentative _r2 existe déjà : choisir _r3 sans toucher au témoin.
    fs::create_dir(sortie.join("lots").join(format!("{base}_r2"))).unwrap();
    let temoin = sortie.join("lots").join(format!("{base}_r2/temoin"));
    fs::write(&temoin, "conserver").unwrap();
    let reprise = pilote(&fichier, &sortie, &[("a", &faux)])
        .args(["--reprendre", "--reprendre-incomplets"])
        .output()
        .unwrap();
    verifier_code(&reprise, 0);
    let doc = index(&sortie);
    assert_eq!(doc["lots"][0]["id"], format!("{base}_r3"));
    assert_eq!(doc["lots"][0]["statut"], "ok");
    assert_eq!(doc["taches"][0]["lot"], format!("{base}_r3"));
    assert_eq!(fs::read(&ancien_log).unwrap(), contenu_log);
    assert_eq!(fs::read(&ancien_manifest).unwrap(), contenu_manifest);
    assert_eq!(fs::read_to_string(&temoin).unwrap(), "conserver");
    let nouvelle = pilote(&fichier, &sortie, &[("a", &faux)])
        .arg("--reprendre")
        .output()
        .unwrap();
    verifier_code(&nouvelle, 0);
    assert_eq!(index(&sortie)["lots"][0]["statut"], "deja_fait");
    // Un changement de paramètres ne doit jamais réutiliser un ancien COMPLETE.
    t.csv("ell,beta,bord,graine,nor\n2,2.5,cold,1,3\n");
    let incompatible = pilote(&fichier, &sortie, &[("a", &faux)])
        .arg("--reprendre")
        .output()
        .unwrap();
    verifier_code(&incompatible, 2);
}

#[test]
#[cfg(unix)]
fn cellules_parametres_et_correspondance_des_chaines() {
    if !python_disponible() {
        return;
    }
    let t = Temp::nouveau("cellules");
    let faux = faux_kvu(&t, "faux");
    let fichier = t.csv("ell,beta,bord,graine,therm,nmeas,skip,nor,options,profil_tous\n2,2.5,cold,1,0,2,0,0,compat-prod,0\n2,2.5,flux,2,0,2,0,0,compat-prod,0\n2,2.5,gauge,3,,,,,,\n2,2.5,haar,4,0,2,0,0,compat-prod,0\n");
    let sortie = t.joindre("sortie");
    let resultat = pilote(&fichier, &sortie, &[("a", &faux)]).output().unwrap();
    verifier_code(&resultat, 0);
    let doc = index(&sortie);
    let lots = doc["lots"].as_array().unwrap();
    assert_eq!(lots.len(), 2);
    let ids: HashSet<_> = lots.iter().map(|l| l["id"].as_str().unwrap()).collect();
    assert_eq!(ids.len(), 2);
    let groupe = lots.iter().find(|l| l["k"] == 3).unwrap();
    let args = groupe["commande"].as_array().unwrap();
    let position = args.iter().position(|v| v == "--chaines").unwrap();
    assert_eq!(args[position + 1], "cold:1,flux:2,haar:4");
    for (i, graine) in [1, 2, 4].iter().enumerate() {
        let tache = doc["taches"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["graine"] == *graine)
            .unwrap();
        assert_eq!(tache["lot"], groupe["id"]);
        assert_eq!(tache["chaine"], i);
        assert!(sortie
            .join(groupe["sortie"].as_str().unwrap())
            .join(format!("c{i:03}/mesures.csv"))
            .exists());
    }
}

#[test]
#[cfg(unix)]
fn binaire_absent_et_manifest_invalide() {
    if !python_disponible() {
        return;
    }
    let t = Temp::nouveau("fichiers");
    let fichier = t.csv("ell,beta,bord,graine\n2,2.5,cold,0\n");
    let sortie = t.joindre("absent");
    let resultat = pilote(&fichier, &sortie, &[("a", &t.joindre("absent_kvu"))])
        .output()
        .unwrap();
    verifier_code(&resultat, 1);
    assert_eq!(index(&sortie)["lots"][0]["statut"], "echec");
    let faux = faux_kvu(&t, "faux");
    let sortie = t.joindre("manifest_invalide");
    let resultat = pilote(&fichier, &sortie, &[("a", &faux)])
        .env("FAUX_MANIFEST_INVALIDE", "1")
        .output()
        .unwrap();
    verifier_code(&resultat, 0);
    assert_eq!(index(&sortie)["lots"][0]["statut"], "ok");
    assert_eq!(index(&sortie)["lots"][0]["verrou"], "non_determine");
}

#[test]
#[cfg(target_os = "linux")]
fn interruption_attend_la_fin_du_faux_processus() {
    if !python_disponible() {
        return;
    }
    let t = Temp::nouveau("interruption");
    let faux = faux_kvu(&t, "faux");
    let fichier = t.csv("ell,beta,bord,graine\n2,2.5,cold,0\n");
    let sortie = t.joindre("sortie");
    let pid = t.joindre("pid");
    let enfant = pilote(&fichier, &sortie, &[("a", &faux)])
        .env("FAUX_SOMMEIL", "60")
        .env("FAUX_PID", &pid)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    // La sentinelle garde l'enfant même si une assertion échoue.
    struct Enfant(std::process::Child);
    impl Drop for Enfant {
        fn drop(&mut self) {
            if self.0.try_wait().ok().flatten().is_some() {
                return;
            }
            // SIGTERM donne au pilote le temps de récolter son propre enfant.
            let _ = Command::new("kill")
                .args(["-TERM", &self.0.id().to_string()])
                .status();
            let _ = self.0.wait();
        }
    }
    let mut garde = Enfant(enfant);
    let limite = Instant::now() + Duration::from_secs(15);
    while !pid.exists() && Instant::now() < limite {
        if garde.0.try_wait().unwrap().is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(pid.exists(), "le faux processus doit démarrer");
    // Défaut d'instabilité corrigé : un fichier lu vide donnait `/proc/` (qui existe toujours) et le faux message « faux processus encore vivant ».
    // Le numéro est lu en entier (le faux l'écrit de façon atomique) ; un fichier incomplet échoue avec son vrai motif.
    let pid_faux: u32 = fs::read_to_string(&pid)
        .unwrap()
        .trim()
        .parse()
        .expect("le fichier pid doit contenir le numéro complet du faux processus");
    let signal = Command::new("kill")
        .args(["-INT", &garde.0.id().to_string()])
        .status()
        .unwrap();
    assert!(signal.success());
    let statut = garde.0.wait().unwrap();
    assert_eq!(statut.code(), Some(130));
    assert!(
        !Path::new("/proc").join(pid_faux.to_string()).exists(),
        "faux processus encore vivant"
    );
    assert_eq!(index(&sortie)["lots"][0]["statut"], "echec");
}

#[test]
fn commande_distante_citee_sans_connexion() {
    if !python_disponible() {
        return;
    }
    // Import sans cache ni appel externe : on inspecte seulement la liste d'arguments.
    let code = r#"
import pathlib
import runpy
import shlex
from types import SimpleNamespace
m = runpy.run_path('scripts/pilote_boites.py', run_name='pilote_test')
t = dict(ell=2, beta=2.5, bord='cold', graine=1, therm=0, nmeas=1, skip=1, nor=0, options='ref-k1', profil_tous=0)
lot = dict(id='L02_b2.5_000', membres=[t])
binaire = "/opt/mon kvu'$(interdit);"
cible = dict(type='ssh', hote='hote-test', kvu=binaire)
a = SimpleNamespace(rep_distant="~/mes lots'$(interdit)")
cmd, distant = m['commande'](lot, cible, pathlib.Path('/tmp/sortie'), 'session-test', a)
assert cmd[:6] == ['ssh', '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=15', 'hote-test']
parent, sep, lancement = cmd[6].partition(' && exec ')
assert sep
assert shlex.split(parent) == ['mkdir', '-p', '--', "$HOME/mes lots'$(interdit)/session-test"]
args = shlex.split(lancement)
assert args[:3] == [binaire, 'su2', 'boite']
assert args[-2:] == ['--outdir', "$HOME/mes lots'$(interdit)/session-test/L02_b2.5_000"]
assert distant == "~/mes lots'$(interdit)/session-test/L02_b2.5_000"
assert args[args.index('--chaines') + 1] == 'cold:1'
"#;
    let sortie = Command::new("python3")
        .current_dir(racine())
        .args(["-I", "-c", code])
        .output()
        .unwrap();
    verifier_code(&sortie, 0);
}

/// Défaut trouvé à l'essai réel (G-DIR-PILOTE-C, 1re passe) : `rsync` (arguments protégés) ne passe par aucun shell distant, donc le chemin
/// du lot ne doit contenir ni `$HOME` ni guillemets ; `~/…` devient un chemin relatif au répertoire personnel distant.
#[test]
fn chemin_de_rapatriement_sans_shell_distant() {
    if !python_disponible() {
        return;
    }
    let code = r#"
import runpy
m = runpy.run_path('scripts/pilote_boites.py', run_name='pilote_test')
assert m['chemin_rsync']('~/kvu_pilote/s_1/L04_b2.5_000') == 'kvu_pilote/s_1/L04_b2.5_000'
assert m['chemin_rsync']("/data/mes lots/s_1/L04_b2.5_000") == "/data/mes lots/s_1/L04_b2.5_000"
assert '$' not in m['chemin_rsync']('~/x') and '"' not in m['chemin_rsync']('~/x')
"#;
    let sortie = Command::new("python3")
        .current_dir(racine())
        .args(["-I", "-c", code])
        .output()
        .unwrap();
    verifier_code(&sortie, 0);
}

// ---------------------------------------------------------------------------------------------------------------------------
// Verrou GPU par cible (défaut trouvé à l'essai réel de G-DIR-PILOTE-D le 25/10 : un lot ssh parti 7 s AVANT la libération du verrou du
// gamer, parce qu'un `kvu` lancé à distance sans kvu.toml prend le verrou par défaut et non celui du poste).
// ---------------------------------------------------------------------------------------------------------------------------

fn flock_disponible() -> bool {
    match Command::new("flock").arg("--version").output() {
        Ok(sortie) if sortie.status.success() => true,
        _ => {
            eprintln!("SKIP : flock est absent");
            false
        }
    }
}

/// Un autre processus qui tient le verrou (flock BSD, comme `flock(1)`) pendant `duree_s` secondes ; il note l'instant de
/// libération dans `trace` AVANT de sortir. Tué à la destruction.
struct Tenant(Child);

impl Tenant {
    fn tenir(verrou: &Path, duree_s: f64, trace: Option<&Path>) -> Self {
        let mut commande = Command::new("python3");
        commande
            .args([
                "-I",
                "-c",
                "import fcntl, sys, time\n\
                 f = open(sys.argv[1], 'w')\n\
                 fcntl.flock(f, fcntl.LOCK_EX)\n\
                 print('tenu', flush=True)\n\
                 time.sleep(float(sys.argv[2]))\n\
                 if len(sys.argv) > 3:\n    open(sys.argv[3], 'w').write(repr(time.time()))\n",
            ])
            .arg(verrou)
            .arg(duree_s.to_string());
        if let Some(trace) = trace {
            commande.arg(trace);
        }
        let mut enfant = commande.stdout(Stdio::piped()).spawn().unwrap();
        let mut ligne = String::new();
        BufReader::new(enfant.stdout.as_mut().unwrap())
            .read_line(&mut ligne)
            .unwrap();
        assert_eq!(ligne.trim(), "tenu", "le tenant n'a pas pris le verrou");
        Self(enfant)
    }
}

impl Drop for Tenant {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn manifeste_du_lot(sortie: &Path, lot: &Value) -> Value {
    let chemin = sortie
        .join(lot["sortie"].as_str().unwrap())
        .join("manifest.json");
    serde_json::from_slice(&fs::read(chemin).unwrap()).unwrap()
}

#[test]
fn verrou_declare_obligatoire_pour_ssh_et_chemin_valide() {
    if !python_disponible() {
        return;
    }
    let t = Temp::nouveau("verrou_decl");
    let fichier = t.csv("ell,beta,bord,graine\n2,2.5,cold,0\n");
    let sortie = t.joindre("sortie");
    let brut = |args: &[&str]| {
        Command::new("python3")
            .current_dir(racine())
            .args(["-I", "scripts/pilote_boites.py"])
            .arg(&fichier)
            .arg("--sortie")
            .arg(&sortie)
            .args(["--cible", "g=ssh:hote-test:/x/kvu", "--dry-run"])
            .args(args)
            .output()
            .unwrap()
    };
    let cas_refuses: [(&[&str], &str); 5] = [
        (&[], "verrou GPU non déclaré"),
        (&["--verrou", "g=relatif/gpu.lock"], "chemin absolu"),
        (&["--verrou", "inconnue=/tmp/x.lock"], "cible inconnue"),
        (
            &["--verrou", "g=/tmp/a.lock", "--verrou", "g=aucun"],
            "répétée",
        ),
        (&["--verrou", "g="], "chemin absolu"),
    ];
    for (args, mot) in cas_refuses {
        let resultat = brut(args);
        verifier_code(&resultat, 2);
        let erreur = String::from_utf8_lossy(&resultat.stderr);
        assert!(erreur.contains(mot), "{args:?} : {erreur}");
        assert!(!sortie.exists());
    }
    let sans_declaration = String::from_utf8_lossy(&brut(&[]).stderr).into_owned();
    assert!(
        sans_declaration.contains("--verrou g=CHEMIN"),
        "{sans_declaration}"
    );
    for valeur in ["g=/tmp/gamer_gpu.lock", "g=~/verrous/gpu.lock", "g=aucun"] {
        let resultat = brut(&["--verrou", valeur]);
        verifier_code(&resultat, 0);
        assert!(String::from_utf8_lossy(&resultat.stdout).contains("PLAN OK"));
    }
    assert!(!sortie.exists(), "--dry-run n'écrit rien");
}

#[test]
fn commande_avec_verrou_citee_et_ordonnee() {
    if !python_disponible() {
        return;
    }
    let code = r#"
import pathlib
import runpy
import shlex
from types import SimpleNamespace
m = runpy.run_path('scripts/pilote_boites.py', run_name='pilote_test')
t = dict(ell=2, beta=2.5, bord='cold', graine=1, therm=0, nmeas=1, skip=1, nor=0, options='ref-k1', profil_tous=0)
lot = dict(id='L02_b2.5_000', membres=[t])
a = SimpleNamespace(rep_distant='~/lots', attente_verrou_s=7200.0)
sortie = pathlib.Path('/tmp/sortie')
# ssh : flock AVANT kvu, chemin du verrou cité (aucune injection)
verrou = "/tmp/mon verrou'$(interdit);"
cible = dict(type='ssh', hote='hote-test', kvu='/opt/kvu', verrou=verrou)
cmd, _ = m['commande'](lot, cible, sortie, 'sess', a)
parent, sep, lancement = cmd[6].partition(' && exec ')
assert sep
args = shlex.split(lancement)
assert args[:6] == ['flock', '-E', '75', '-w', '7200', verrou], args[:6]
assert args[6:9] == ['/opt/kvu', 'su2', 'boite'], args[6:9]
# ssh : verrou sous ~/ développé par le shell distant
cible['verrou'] = '~/verrous/gpu.lock'
cmd, _ = m['commande'](lot, cible, sortie, 'sess', a)
parent, sep, lancement = cmd[6].partition(' && exec ')
assert lancement.startswith('flock -E 75 -w 7200 "$HOME"/verrous/gpu.lock /opt/kvu su2 boite '), lancement
# local : même enveloppe, sans shell
cible = dict(type='local', hote=None, kvu='/opt/kvu', verrou='/tmp/gamer_gpu.lock')
cmd, _ = m['commande'](lot, cible, sortie, 'sess', a)
assert cmd[:9] == ['flock', '-E', '75', '-w', '7200', '/tmp/gamer_gpu.lock', '/opt/kvu', 'su2', 'boite'], cmd[:9]
# délai décimal conservé
a.attente_verrou_s = 0.5
cmd, _ = m['commande'](lot, cible, sortie, 'sess', a)
assert cmd[3:5] == ['-w', '0.5'], cmd[:6]
# sans verrou : aucune enveloppe (kvu prend lui-même celui de sa machine)
cible['verrou'] = None
cmd, _ = m['commande'](lot, cible, sortie, 'sess', a)
assert cmd[0] == '/opt/kvu' and 'flock' not in cmd
assert m['CODE_VERROU'] == 75
"#;
    let sortie = Command::new("python3")
        .current_dir(racine())
        .args(["-I", "-c", code])
        .output()
        .unwrap();
    verifier_code(&sortie, 0);
}

#[test]
#[cfg(unix)]
fn verrou_local_le_lot_attend_la_liberation() {
    if !python_disponible() || !flock_disponible() {
        return;
    }
    let t = Temp::nouveau("verrou_attente");
    let faux = faux_kvu(&t, "faux_v");
    let verrou = t.joindre("gpu.lock");
    let liberation = t.joindre("liberation.txt");
    let tenant = Tenant::tenir(&verrou, 1.5, Some(&liberation));
    let fichier = t.csv("ell,beta,bord,graine\n2,2.5,cold,7\n");
    let sortie = t.joindre("sortie");
    let resultat = pilote(&fichier, &sortie, &[("v", &faux)])
        .args(["--verrou", &format!("v={}", verrou.display())])
        .output()
        .unwrap();
    drop(tenant);
    verifier_code(&resultat, 0);
    let doc = index(&sortie);
    let lot = &doc["lots"][0];
    assert_eq!(lot["statut"], "ok");
    assert_eq!(doc["cibles"]["v"]["verrou"], verrou.to_str().unwrap());
    let debut = manifeste_du_lot(&sortie, lot)["debut_unix"]
        .as_f64()
        .unwrap();
    let liberee: f64 = fs::read_to_string(&liberation)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert!(
        debut >= liberee - 0.05,
        "le lot a démarré {:.3} s AVANT la libération du verrou",
        liberee - debut
    );
}

#[test]
#[cfg(unix)]
fn verrou_local_delai_depasse_est_un_echec_lisible() {
    if !python_disponible() || !flock_disponible() {
        return;
    }
    let t = Temp::nouveau("verrou_delai");
    let faux = faux_kvu(&t, "faux_w");
    let verrou = t.joindre("gpu.lock");
    let _tenant = Tenant::tenir(&verrou, 60.0, None);
    let fichier = t.csv("ell,beta,bord,graine\n2,2.5,cold,8\n");
    let sortie = t.joindre("sortie");
    let debut = Instant::now();
    let resultat = pilote(&fichier, &sortie, &[("v", &faux)])
        .args([
            "--verrou",
            &format!("v={}", verrou.display()),
            "--attente-verrou-s",
            "1",
        ])
        .output()
        .unwrap();
    assert!(
        debut.elapsed() < Duration::from_secs(30),
        "le pilote ne doit pas attendre au-delà du délai demandé"
    );
    verifier_code(&resultat, 1);
    let doc = index(&sortie);
    let lot = &doc["lots"][0];
    assert_eq!(lot["statut"], "echec");
    assert_eq!(lot["rc"], 75);
    assert!(
        lot["erreur"].as_str().unwrap().contains("verrou GPU"),
        "{lot}"
    );
    assert!(
        !sortie.join(lot["sortie"].as_str().unwrap()).exists(),
        "le faux kvu n'a pas dû démarrer"
    );
}
