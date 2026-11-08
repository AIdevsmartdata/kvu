//! Tests de la file de jobs (tâche T1) : `construire_commande` golden, transitions atomiques,
//! deux workers concurrents, admissibilité (RAM, SSD, moniteur), exécution, ssh distant simulé,
//! temps civil, espace libre, et les sous-commandes `queue`.

use kvu::commun::{Erreur, Machine};
use kvu::machine::cli::principal_machine;
use kvu::machine::detection::{GpuDetecte, ProfilMachine};
use kvu::machine::file::*;
use kvu::machine::politique::{politique_avec, Politique};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Barrier, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------------
// Aides
// ---------------------------------------------------------------------------------------------

/// Dossier temporaire supprimé à la fin du test.
struct Temp(PathBuf);

impl Temp {
    fn nouveau(nom: &str) -> Temp {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let d = std::env::temp_dir().join(format!("kvu_file_{nom}_{}_{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Temp(d)
    }
    fn chemin(&self) -> &Path {
        &self.0
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn profil(machine: Machine) -> ProfilMachine {
    ProfilMachine {
        machine,
        hote: "h".into(),
        cpu_modele: "c".into(),
        fils_logiques: 8,
        ram_totale_go: 32.0,
        ram_disponible_go: 20.0,
        gpus: Vec::<GpuDetecte>::new(),
        versions: vec![],
        surcharge_refusee: None,
    }
}

fn pol(machine: Machine) -> Politique {
    politique_avec(&profil(machine), &|_| None, Path::new("/home/x"))
}

/// Politique de test : aucune ressource ne refuse (RAM, SSD, moniteur), commandes sans nice spécial.
fn pol_libre(t: &Temp) -> Politique {
    let mut p = pol(Machine::Inconnue);
    p.ram_min_go = 0.0;
    p.seuil_ssd_go = 0.0;
    p.alerte_moniteur = None;
    p.mem_max = None;
    p.file = t.chemin().join("file");
    p.verrou_gpu = t.chemin().join("gpu.lock");
    p
}

fn job(cmd: &[&str]) -> Job {
    Job::nouveau(Machine::Inconnue, cmd.iter().map(|s| s.to_string()).collect())
}

fn sh(script: &str) -> Job {
    job(&["sh", "-c", script])
}

fn ouvrir(t: &Temp) -> File {
    File::ouvrir(&t.chemin().join("file")).unwrap()
}

fn v(s: &[&str]) -> Vec<String> {
    s.iter().map(|x| x.to_string()).collect()
}

// ---------------------------------------------------------------------------------------------
// construire_commande : golden
// ---------------------------------------------------------------------------------------------

#[test]
fn commande_golden_acemagic() {
    let p = pol(Machine::Acemagic);
    assert_eq!(
        construire_commande(&job(&["echo", "hi"]), &p),
        v(&["systemd-run", "--scope", "-p", "MemoryMax=24G", "--quiet", "nice", "-n", "10", "echo", "hi"])
    );
    let mut j = job(&["kevinotron", "su2", "gen"]);
    j.gpu = true;
    assert_eq!(
        construire_commande(&j, &p),
        v(&[
            "systemd-run",
            "--scope",
            "-p",
            "MemoryMax=24G",
            "--quiet",
            "nice",
            "-n",
            "10",
            "flock",
            "-w",
            "7200",
            "/tmp/kevinotron_gpu.lock",
            "kevinotron",
            "su2",
            "gen"
        ])
    );
}

#[test]
fn commande_golden_gamer_taskset_nice5_et_memoire() {
    let p = pol(Machine::Gamer);
    let mut j = job(&["kevinotron", "su2", "gen"]);
    j.gpu = true;
    assert_eq!(
        construire_commande(&j, &p),
        v(&[
            "systemd-run",
            "--scope",
            "-p",
            "MemoryMax=12G",
            "--quiet",
            "nice",
            "-n",
            "5",
            "taskset",
            "-c",
            "12-19",
            "flock",
            "-w",
            "7200",
            "/tmp/gamer_gpu.lock",
            "kevinotron",
            "su2",
            "gen"
        ])
    );
    // mem donné : il remplace le plafond de la politique.
    j.mem_max = Some("8G".into());
    assert_eq!(
        construire_commande(&j, &p),
        v(&[
            "systemd-run",
            "--scope",
            "-p",
            "MemoryMax=8G",
            "--quiet",
            "nice",
            "-n",
            "5",
            "taskset",
            "-c",
            "12-19",
            "flock",
            "-w",
            "7200",
            "/tmp/gamer_gpu.lock",
            "kevinotron",
            "su2",
            "gen"
        ])
    );
    // CPU seul : jamais de flock, mais toujours taskset sur le gamer (et le plafond mémoire).
    assert_eq!(
        construire_commande(&job(&["make"]), &p),
        v(&["systemd-run", "--scope", "-p", "MemoryMax=12G", "--quiet", "nice", "-n", "5", "taskset", "-c", "12-19", "make"])
    );
}

/// Vérificateur du 04/10 : la taille DISQUE estimée servait à décider du plafond RAM. Désormais tout job a un
/// MemoryMax (celui du job, sinon celui de la politique), quelle que soit sa sortie estimée.
#[test]
fn commande_systemd_run_pour_tout_job() {
    let p = pol(Machine::Acemagic);
    let mut j = job(&["x"]);
    for sortie in [0.0, 4.0, 4.1, 100.0] {
        j.sortie_estimee_go = sortie;
        assert_eq!(construire_commande(&j, &p)[3], "MemoryMax=24G", "sortie estimée {sortie} Go");
    }
    j.sortie_estimee_go = 4.1;
    assert_eq!(
        construire_commande(&j, &p),
        v(&[
            "systemd-run",
            "--scope",
            "-p",
            "MemoryMax=24G",
            "--quiet",
            "nice",
            "-n",
            "10",
            "x"
        ])
    );
    // Le mem du job l'emporte sur celui de la politique.
    j.mem_max = Some("6G".into());
    assert_eq!(construire_commande(&j, &p)[3], "MemoryMax=6G");
    // Ni le job ni la politique n'ont de plafond (politique de test) : pas de systemd-run.
    let mut p2 = p.clone();
    p2.mem_max = None;
    j.mem_max = None;
    assert_eq!(construire_commande(&j, &p2)[0], "nice");
}

#[test]
fn commande_sans_flock_si_verrou_deja_tenu_par_un_ancetre() {
    let p = pol(Machine::Acemagic);
    let mut j = job(&["k"]);
    j.gpu = true;
    assert!(construire_commande(&j, &p).contains(&"flock".to_string()));
    assert_eq!(
        construire_commande_ex(&j, &p, true),
        v(&["systemd-run", "--scope", "-p", "MemoryMax=24G", "--quiet", "nice", "-n", "10", "k"])
    );
    assert_eq!(construire_commande_ex(&j, &p, false), construire_commande(&j, &p));
}

// ---------------------------------------------------------------------------------------------
// Temps et espace libre
// ---------------------------------------------------------------------------------------------

#[test]
fn temps_civil_utc() {
    assert_eq!(horodatage_iso(0.0), "1970-01-01T00:00:00Z");
    assert_eq!(horodatage_iso(86_399.9), "1970-01-01T23:59:59Z");
    assert_eq!(horodatage_iso(951_782_400.0), "2000-02-29T00:00:00Z"); // année bissextile
    assert_eq!(horodatage_iso(951_782_400.0 + 86_400.0), "2000-03-01T00:00:00Z");
    assert_eq!(horodatage_iso(1_759_536_000.0), "2025-10-04T00:00:00Z");
    assert_eq!(horodatage_iso(4_102_444_800.0), "2100-01-01T00:00:00Z");
    assert_eq!(prefixe_id(1_759_579_199.0), "20251004-115959");
}

#[test]
fn espace_libre_comme_df() {
    let libre = libre_go(Path::new("/tmp")).expect("statvfs /tmp");
    let df = Command::new("df")
        .args(["--output=avail", "-k", "/tmp"])
        .output()
        .unwrap();
    let ko: f64 = String::from_utf8_lossy(&df.stdout)
        .lines()
        .nth(1)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    let attendu = ko * 1024.0 / (1u64 << 30) as f64;
    assert!(
        (libre - attendu).abs() < 0.5,
        "statvfs {libre} Gio contre df {attendu} Gio"
    );
    // Chemin inexistant : on remonte au premier ancêtre existant ; chemin vide : dossier courant.
    let sous = libre_go(Path::new("/tmp/kvu_nexiste_pas/a/b/c")).unwrap();
    assert!((sous - libre).abs() < 0.5);
    assert!(libre_go(Path::new("")).is_some());
}

// ---------------------------------------------------------------------------------------------
// Admissibilité pure
// ---------------------------------------------------------------------------------------------

#[test]
fn raison_refus_matrice() {
    let mut p = pol(Machine::Acemagic); // ram 25, ssd 100
    let mut j = job(&["x"]);
    assert_eq!(raison_refus(&j, &p, 30.0, None, Some(500.0)), None);
    // RAM
    let r = raison_refus(&j, &p, 24.9, None, Some(500.0)).unwrap();
    assert!(
        r.starts_with("[ram]") && r.contains("24.9") && r.contains("25.0"),
        "{r}"
    );
    // Alerte récente (< 10 min) refuse ; ancienne (≥ 10 min) accepte.
    assert!(raison_refus(&j, &p, 30.0, Some(599.0), Some(500.0))
        .unwrap()
        .starts_with("[alerte]"));
    assert_eq!(raison_refus(&j, &p, 30.0, Some(600.0), Some(500.0)), None);
    // SSD : libre − sortie ≥ seuil.
    j.sortie_estimee_go = 400.0;
    assert_eq!(
        raison_refus(&j, &p, 30.0, None, Some(500.0)),
        None,
        "500 − 400 = 100 ≥ 100"
    );
    j.sortie_estimee_go = 400.1;
    assert!(raison_refus(&j, &p, 30.0, None, Some(500.0))
        .unwrap()
        .starts_with("[ssd]"));
    // Espace illisible : refus par prudence.
    assert!(raison_refus(&j, &p, 30.0, None, None).unwrap().contains("illisible"));
    // L'ordre : la RAM d'abord.
    p.ram_min_go = 1e9;
    assert!(raison_refus(&j, &p, 30.0, Some(1.0), Some(1.0))
        .unwrap()
        .starts_with("[ram]"));
}

// ---------------------------------------------------------------------------------------------
// File : ajout, liste, annulation
// ---------------------------------------------------------------------------------------------

#[test]
fn ouvrir_cree_larborescence() {
    let t = Temp::nouveau("arbo");
    let f = ouvrir(&t);
    for d in ["attente", "en_cours", "fini", "echec", "annule", "journaux"] {
        assert!(f.racine.join(d).is_dir(), "{d}");
    }
    // Réouverture : idempotente.
    assert!(File::ouvrir(&f.racine).is_ok());
}

#[test]
fn ajout_et_lecture_a_lidentique() {
    let t = Temp::nouveau("ajout");
    let f = ouvrir(&t);
    let mut j = job(&["kevinotron", "su2", "gen", "--L", "16"]);
    j.gpu = true;
    j.mem_max = Some("8G".into());
    j.fils = 4;
    j.priorite = -2;
    j.sortie_estimee_go = 2.5;
    j.repertoire = PathBuf::from("/tmp");
    let id = f.ajouter(j.clone()).unwrap();
    let l = f.lister().unwrap();
    assert_eq!(l.len(), 1);
    let lu = &l[0];
    assert_eq!(lu.id, id);
    assert_eq!(lu.etat, EtatJob::Attente);
    assert_eq!(lu.commande, j.commande);
    assert_eq!((lu.gpu, lu.fils, lu.priorite), (true, 4, -2));
    assert_eq!(lu.mem_max.as_deref(), Some("8G"));
    assert_eq!(lu.sortie_estimee_go, 2.5);
    assert_eq!(lu.repertoire, Path::new("/tmp"));
    assert!(lu.cree_unix > 1.7e9);
    assert!(f.racine.join("attente").join(format!("{id}.json")).exists());
    // Pas de fichier temporaire résiduel.
    let residus: Vec<_> = std::fs::read_dir(f.racine.join("attente"))
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with('.'))
        .collect();
    assert!(residus.is_empty());
}

#[test]
fn identifiants_croissants_et_formats() {
    let t = Temp::nouveau("ids");
    let f = ouvrir(&t);
    let ids: Vec<String> = (0..12).map(|_| f.ajouter(job(&["true"])).unwrap()).collect();
    let mut tries = ids.clone();
    tries.sort();
    assert_eq!(ids, tries, "ordre lexicographique = ordre d'arrivée");
    assert_eq!(ids.iter().collect::<HashSet<_>>().len(), 12);
    for id in &ids {
        let parts: Vec<&str> = id.split('-').collect();
        assert_eq!(parts.len(), 4, "{id}");
        assert!(
            parts[0].len() == 8 && parts[1].len() == 6 && parts[2] == "inconnue" && parts[3].len() == 4,
            "{id}"
        );
    }
    // Date imposée : l'identifiant la reflète.
    let mut j = Job::nouveau(Machine::Gamer, v(&["x"]));
    j.cree_unix = 1_759_536_000.0;
    assert_eq!(f.ajouter(j).unwrap(), "20251004-000000-gamer-0001");
}

#[test]
fn identifiant_impose_collision_et_invalide() {
    let t = Temp::nouveau("idimpose");
    let f = ouvrir(&t);
    let mut j = job(&["true"]);
    j.id = "mon-job".into();
    assert_eq!(f.ajouter(j.clone()).unwrap(), "mon-job");
    // Jamais d'écrasement, ni en attente ni dans un autre état.
    assert!(matches!(f.ajouter(j.clone()), Err(Erreur::Config(_))));
    f.annuler("mon-job").unwrap();
    assert!(
        matches!(f.ajouter(j.clone()), Err(Erreur::Config(_))),
        "même annulé, l'id reste pris"
    );
    for mauvais in ["../x", "a/b", ".cache"] {
        let mut k = job(&["true"]);
        k.id = mauvais.into();
        assert!(matches!(f.ajouter(k), Err(Erreur::Config(_))), "{mauvais}");
    }
    assert!(
        matches!(f.ajouter(job(&[])), Err(Erreur::Config(_))),
        "job sans commande"
    );
}

#[test]
fn ajouts_concurrents_donnent_des_ids_distincts() {
    let t = Temp::nouveau("ajoutconc");
    let f = Arc::new(ouvrir(&t));
    let barriere = Arc::new(Barrier::new(8));
    let hs: Vec<_> = (0..8)
        .map(|_| {
            let (f, b) = (f.clone(), barriere.clone());
            std::thread::spawn(move || {
                b.wait();
                (0..10).map(|_| f.ajouter(job(&["true"])).unwrap()).collect::<Vec<_>>()
            })
        })
        .collect();
    let tous: Vec<String> = hs.into_iter().flat_map(|h| h.join().unwrap()).collect();
    assert_eq!(tous.len(), 80);
    assert_eq!(tous.iter().collect::<HashSet<_>>().len(), 80, "ids dupliqués");
    assert_eq!(f.lister().unwrap().len(), 80);
    assert_eq!(f.compter(EtatJob::Attente), 80);
}

#[test]
fn fichiers_illisibles_ignores_mais_signales() {
    let t = Temp::nouveau("invalide");
    let f = ouvrir(&t);
    f.ajouter(job(&["true"])).unwrap();
    std::fs::write(f.racine.join("attente/casse.json"), "{pas du json").unwrap();
    std::fs::write(f.racine.join("attente/.cache.json"), "caché").unwrap();
    std::fs::write(f.racine.join("attente/notes.txt"), "autre").unwrap();
    assert_eq!(f.lister().unwrap().len(), 1);
    assert_eq!(f.invalides(), vec![f.racine.join("attente/casse.json")]);
    // Le worker n'est pas bloqué par le fichier cassé.
    assert!(f.prendre_suivant(&pol_libre(&t)).unwrap().is_some());
}

#[test]
fn annulation() {
    let t = Temp::nouveau("annuler");
    let f = ouvrir(&t);
    let id = f.ajouter(job(&["true"])).unwrap();
    f.annuler(&id).unwrap();
    assert_eq!(f.lister_etat(EtatJob::Annule).unwrap().len(), 1);
    assert!(f.lister_etat(EtatJob::Attente).unwrap().is_empty());
    // Deuxième annulation, et id inconnu : erreurs d'usage.
    assert!(matches!(f.annuler(&id), Err(Erreur::Config(m)) if m.contains("plus en attente")));
    assert!(matches!(f.annuler("inconnu"), Err(Erreur::Config(m)) if m.contains("inconnu")));
    // Un job annulé n'est jamais pris.
    assert!(f.prendre_suivant(&pol_libre(&t)).unwrap().is_none());
    // Un job EN COURS n'est pas tué.
    let id2 = f.ajouter(job(&["true"])).unwrap();
    let pris = f.prendre_suivant(&pol_libre(&t)).unwrap().unwrap();
    assert_eq!(pris.id, id2);
    assert!(matches!(f.annuler(&id2), Err(Erreur::Config(m)) if m.contains("en cours")));
}

// ---------------------------------------------------------------------------------------------
// prendre_suivant : ordre, ressources, concurrence
// ---------------------------------------------------------------------------------------------

#[test]
fn ordre_priorite_decroissante_puis_id_croissant() {
    let t = Temp::nouveau("ordre");
    let f = ouvrir(&t);
    let mut ids = Vec::new();
    for (nom, prio) in [("a", 0), ("b", 5), ("c", 0), ("d", 5), ("e", -1)] {
        let mut j = job(&["true"]);
        j.id = format!("id-{nom}");
        j.priorite = prio;
        ids.push(f.ajouter(j).unwrap());
    }
    let p = pol_libre(&t);
    let ordre: Vec<String> = std::iter::from_fn(|| f.prendre_suivant(&p).unwrap().map(|j| j.id)).collect();
    assert_eq!(ordre, v(&["id-b", "id-d", "id-a", "id-c", "id-e"]));
}

#[test]
fn la_prise_deplace_le_fichier_et_note_le_debut() {
    let t = Temp::nouveau("prise");
    let f = ouvrir(&t);
    let id = f.ajouter(job(&["true"])).unwrap();
    let pris = f.prendre_suivant(&pol_libre(&t)).unwrap().unwrap();
    assert_eq!(pris.etat, EtatJob::EnCours);
    assert!(pris.debut_unix.is_some());
    assert_eq!(pris.journal.as_deref(), Some(f.chemin_journal(&id).as_path()));
    assert!(!f.racine.join("attente").join(format!("{id}.json")).exists());
    let en_cours = f.lister_etat(EtatJob::EnCours).unwrap();
    assert_eq!(en_cours.len(), 1);
    assert!(en_cours[0].debut_unix.is_some());
}

#[test]
fn deux_workers_un_job_execute_une_seule_fois() {
    // 200 rondes : à chaque ronde, 2 workers se disputent le même job.
    for ronde in 0..200 {
        let t = Temp::nouveau(&format!("duel{ronde}"));
        let f = Arc::new(ouvrir(&t));
        f.ajouter(job(&["true"])).unwrap();
        let p = Arc::new(pol_libre(&t));
        let barriere = Arc::new(Barrier::new(2));
        let hs: Vec<_> = (0..2)
            .map(|_| {
                let (f, p, b) = (f.clone(), p.clone(), barriere.clone());
                std::thread::spawn(move || {
                    b.wait();
                    f.prendre_suivant(&p).unwrap().is_some()
                })
            })
            .collect();
        let gagnants = hs.into_iter().map(|h| h.join().unwrap()).filter(|x| *x).count();
        assert_eq!(gagnants, 1, "ronde {ronde} : {gagnants} workers ont pris le même job");
    }
}

#[test]
fn huit_workers_quarante_jobs_chacun_pris_exactement_une_fois() {
    let t = Temp::nouveau("huit");
    let f = Arc::new(ouvrir(&t));
    let attendus: HashSet<String> = (0..40).map(|_| f.ajouter(job(&["true"])).unwrap()).collect();
    let p = Arc::new(pol_libre(&t));
    let barriere = Arc::new(Barrier::new(8));
    let hs: Vec<_> = (0..8)
        .map(|_| {
            let (f, p, b) = (f.clone(), p.clone(), barriere.clone());
            std::thread::spawn(move || {
                b.wait();
                std::iter::from_fn(|| f.prendre_suivant(&p).unwrap().map(|j| j.id)).collect::<Vec<_>>()
            })
        })
        .collect();
    let pris: Vec<String> = hs.into_iter().flat_map(|h| h.join().unwrap()).collect();
    assert_eq!(pris.len(), 40, "tous les jobs pris, aucun en double");
    assert_eq!(pris.into_iter().collect::<HashSet<_>>(), attendus);
    assert_eq!(f.compter(EtatJob::EnCours), 40);
    assert_eq!(f.compter(EtatJob::Attente), 0);
}

#[test]
fn ram_insuffisante_le_job_reste_en_attente_avec_la_raison() {
    let t = Temp::nouveau("ram");
    let f = ouvrir(&t);
    let id = f.ajouter(job(&["true"])).unwrap();
    let mut p = pol_libre(&t);
    p.ram_min_go = 1e9; // simulé : aucune machine n'a 10⁹ Go
    let prise = f.prendre_suivant_detail(&p).unwrap();
    assert!(prise.job.is_none());
    assert_eq!(prise.refus.len(), 1);
    assert!(prise.refus[0].1.starts_with("[ram]"), "{:?}", prise.refus);
    assert_eq!(f.compter(EtatJob::Attente), 1, "le job reste en attente");
    let journal = std::fs::read_to_string(f.chemin_journal(&id)).unwrap();
    assert!(journal.contains("en attente : [ram] RAM disponible"), "{journal}");
}

#[test]
fn ssd_insuffisant_garde_le_job_et_journalise() {
    let t = Temp::nouveau("ssd");
    let f = ouvrir(&t);
    let mut j = job(&["true"]);
    j.sortie_estimee_go = 1e9;
    let id = f.ajouter(j).unwrap();
    let mut p = pol_libre(&t);
    p.seuil_ssd_go = 1.0;
    assert!(f.prendre_suivant(&p).unwrap().is_none());
    assert!(std::fs::read_to_string(f.chemin_journal(&id))
        .unwrap()
        .contains("[ssd] libre"));
    // Un seuil au-dessus du libre refuse aussi un job sans sortie.
    let t2 = Temp::nouveau("ssd2");
    let f2 = ouvrir(&t2);
    f2.ajouter(job(&["true"])).unwrap();
    let mut p2 = pol_libre(&t2);
    p2.seuil_ssd_go = 1e9;
    assert!(f2.prendre_suivant(&p2).unwrap().is_none());
}

#[test]
fn alerte_du_moniteur_recente_refuse_ancienne_accepte() {
    let t = Temp::nouveau("alerte");
    let f = ouvrir(&t);
    let id = f.ajouter(job(&["true"])).unwrap();
    let alerte = t.chemin().join("ALERTE.txt");
    std::fs::write(&alerte, "charge trop haute").unwrap();
    let mut p = pol_libre(&t);
    p.alerte_moniteur = Some(alerte.clone());
    assert!(f.prendre_suivant(&p).unwrap().is_none());
    assert!(std::fs::read_to_string(f.chemin_journal(&id))
        .unwrap()
        .contains("[alerte]"));
    // Vieillie de 11 minutes : le job passe.
    let vieux = std::time::SystemTime::now() - Duration::from_secs(660);
    std::fs::File::options()
        .write(true)
        .open(&alerte)
        .unwrap()
        .set_modified(vieux)
        .unwrap();
    assert_eq!(f.prendre_suivant(&p).unwrap().map(|j| j.id), Some(id.clone()));
    // Fichier d'alerte configuré mais absent : aucune alerte.
    let t2 = Temp::nouveau("alerte_abs");
    let f2 = ouvrir(&t2);
    f2.ajouter(job(&["true"])).unwrap();
    let mut p2 = pol_libre(&t);
    p2.alerte_moniteur = Some(t.chemin().join("n_existe_pas.txt"));
    assert!(f2.prendre_suivant(&p2).unwrap().is_some());
}

#[test]
fn raison_journalisee_une_seule_fois_par_etiquette() {
    let t = Temp::nouveau("dedup");
    let f = ouvrir(&t);
    let id = f.ajouter(job(&["true"])).unwrap();
    let mut p = pol_libre(&t);
    p.ram_min_go = 1e9;
    for _ in 0..5 {
        assert!(f.prendre_suivant(&p).unwrap().is_none());
    }
    let n = |motif: &str| {
        std::fs::read_to_string(f.chemin_journal(&id))
            .unwrap()
            .matches(motif)
            .count()
    };
    assert_eq!(n("en attente : [ram]"), 1, "pas de spam du journal à chaque sondage");
    // Une AUTRE raison s'ajoute.
    p.ram_min_go = 0.0;
    p.seuil_ssd_go = 1e9;
    assert!(f.prendre_suivant(&p).unwrap().is_none());
    assert_eq!((n("en attente : [ram]"), n("en attente : [ssd]")), (1, 1));
}

#[test]
fn un_gros_job_refuse_ne_bloque_pas_un_petit() {
    let t = Temp::nouveau("backfill");
    let f = ouvrir(&t);
    let mut gros = job(&["true"]);
    gros.id = "gros".into();
    gros.priorite = 10;
    gros.sortie_estimee_go = 1e9;
    let mut petit = job(&["true"]);
    petit.id = "petit".into();
    f.ajouter(gros).unwrap();
    f.ajouter(petit).unwrap();
    let mut p = pol_libre(&t);
    p.seuil_ssd_go = 1.0;
    let prise = f.prendre_suivant_detail(&p).unwrap();
    assert_eq!(prise.job.map(|j| j.id), Some("petit".to_string()));
    assert_eq!(prise.refus.len(), 1);
    assert_eq!(prise.refus[0].0, "gros");
    assert_eq!(f.compter(EtatJob::Attente), 1);
}

// ---------------------------------------------------------------------------------------------
// executer
// ---------------------------------------------------------------------------------------------

#[test]
fn execution_reussie_range_dans_fini_avec_journal() {
    let t = Temp::nouveau("exec_ok");
    let f = ouvrir(&t);
    let p = pol_libre(&t);
    let id = f.ajouter(sh("echo bonjour-sortie; echo bonjour-erreur >&2")).unwrap();
    let pris = f.prendre_suivant(&p).unwrap().unwrap();
    assert_eq!(f.executer(pris, &p).unwrap(), 0);
    let fini = f.lister_etat(EtatJob::Fini).unwrap();
    assert_eq!(fini.len(), 1);
    let j = &fini[0];
    assert_eq!((j.id.as_str(), j.rc, j.etat), (id.as_str(), Some(0), EtatJob::Fini));
    assert!(j.pid.is_some() && j.debut_unix.is_some() && j.fin_unix.unwrap() >= j.debut_unix.unwrap());
    assert!(f.lister_etat(EtatJob::EnCours).unwrap().is_empty());
    let journal = std::fs::read_to_string(f.chemin_journal(&id)).unwrap();
    for motif in [
        "lancement : nice -n 10 sh -c",
        "bonjour-sortie",
        "bonjour-erreur",
        "fin : rc = 0",
    ] {
        assert!(journal.contains(motif), "« {motif} » manque :\n{journal}");
    }
}

#[test]
fn execution_en_echec_range_dans_echec_avec_le_rc() {
    let t = Temp::nouveau("exec_ko");
    let f = ouvrir(&t);
    let p = pol_libre(&t);
    f.ajouter(sh("echo avant; exit 3")).unwrap();
    let pris = f.prendre_suivant(&p).unwrap().unwrap();
    assert_eq!(f.executer(pris, &p).unwrap(), 3);
    let e = f.lister_etat(EtatJob::Echec).unwrap();
    assert_eq!((e.len(), e[0].rc, e[0].etat), (1, Some(3), EtatJob::Echec));
    assert!(f.lister_etat(EtatJob::Fini).unwrap().is_empty());
}

#[test]
fn commande_introuvable_donne_127_sans_planter() {
    let t = Temp::nouveau("exec_127");
    let f = ouvrir(&t);
    let p = pol_libre(&t);
    // Tombe sous `nice`, qui échoue (rc 127) : même résultat que si le programme manquait.
    let id = f.ajouter(job(&["commande-qui-n-existe-pas-kvu"])).unwrap();
    let pris = f.prendre_suivant(&p).unwrap().unwrap();
    assert_eq!(f.executer(pris, &p).unwrap(), 127);
    assert_eq!(f.lister_etat(EtatJob::Echec).unwrap()[0].rc, Some(127));
    assert!(std::fs::read_to_string(f.chemin_journal(&id))
        .unwrap()
        .contains("fin : rc = 127"));
}

#[test]
fn processus_tue_par_un_signal_rend_128_plus_signal() {
    let t = Temp::nouveau("exec_sig");
    let f = ouvrir(&t);
    let p = pol_libre(&t);
    f.ajouter(sh("kill -KILL $$")).unwrap();
    let pris = f.prendre_suivant(&p).unwrap().unwrap();
    assert_eq!(f.executer(pris, &p).unwrap(), 128 + 9);
}

#[test]
fn executer_refuse_un_job_non_pris() {
    let t = Temp::nouveau("exec_nonpris");
    let f = ouvrir(&t);
    let p = pol_libre(&t);
    let id = f.ajouter(job(&["true"])).unwrap();
    let j = f.lister().unwrap().remove(0);
    assert!(matches!(f.executer(j, &p), Err(Erreur::Config(m)) if m.contains("pas été pris")));
    assert_eq!(f.compter(EtatJob::Attente), 1, "{id} intact");
}

#[test]
fn repertoire_et_fils_sont_appliques() {
    let t = Temp::nouveau("exec_env");
    let f = ouvrir(&t);
    let p = pol_libre(&t); // fils_max = 4 (inconnue)
    let travail = t.chemin().join("travail");
    std::fs::create_dir_all(&travail).unwrap();
    let mut j = sh("pwd; echo fils=$RAYON_NUM_THREADS");
    j.repertoire = travail.clone();
    j.fils = 16;
    let id = f.ajouter(j).unwrap();
    let pris = f.prendre_suivant(&p).unwrap().unwrap();
    assert_eq!(f.executer(pris, &p).unwrap(), 0);
    let journal = std::fs::read_to_string(f.chemin_journal(&id)).unwrap();
    assert!(journal.contains(&travail.display().to_string()), "{journal}");
    assert!(journal.contains("fils=4"), "fils plafonné par la politique : {journal}");
}

#[test]
fn boucle_vide_la_file_dans_lordre() {
    let t = Temp::nouveau("boucle");
    let f = ouvrir(&t);
    let p = pol_libre(&t);
    let trace = t.chemin().join("trace.txt");
    for (nom, prio) in [("a", 0), ("b", 9), ("c", 0)] {
        let mut j = sh(&format!("echo {nom} >> {}", trace.display()));
        j.id = format!("job-{nom}");
        j.priorite = prio;
        f.ajouter(j).unwrap();
    }
    let faits = f.boucle(&p, true, Duration::from_millis(10), None).unwrap();
    assert_eq!(
        faits,
        vec![
            ("job-b".to_string(), 0),
            ("job-a".to_string(), 0),
            ("job-c".to_string(), 0)
        ]
    );
    assert_eq!(std::fs::read_to_string(&trace).unwrap(), "b\na\nc\n");
    assert_eq!(f.compter(EtatJob::Fini), 3);
    // `max` borne le nombre de jobs.
    f.ajouter(sh("true")).unwrap();
    f.ajouter(sh("true")).unwrap();
    assert_eq!(f.boucle(&p, true, Duration::from_millis(10), Some(1)).unwrap().len(), 1);
    assert_eq!(f.compter(EtatJob::Attente), 1);
}

#[test]
fn deux_workers_concurrents_executent_chaque_job_une_fois() {
    let t = Temp::nouveau("workers");
    let f = Arc::new(ouvrir(&t));
    let trace = t.chemin().join("trace.txt");
    for i in 0..12 {
        f.ajouter(sh(&format!("echo job{i} >> {}", trace.display()))).unwrap();
    }
    let p = Arc::new(pol_libre(&t));
    let hs: Vec<_> = (0..2)
        .map(|_| {
            let (f, p) = (f.clone(), p.clone());
            std::thread::spawn(move || f.boucle(&p, true, Duration::from_millis(5), None).unwrap())
        })
        .collect();
    let faits: Vec<(String, i32)> = hs.into_iter().flat_map(|h| h.join().unwrap()).collect();
    assert_eq!(faits.len(), 12);
    assert!(faits.iter().all(|(_, rc)| *rc == 0));
    let texte = std::fs::read_to_string(&trace).unwrap();
    let mut lignes: Vec<&str> = texte.lines().collect();
    lignes.sort_unstable();
    lignes.dedup();
    assert_eq!(lignes.len(), 12, "chaque job exactement une fois :\n{texte}");
    assert_eq!(f.compter(EtatJob::Fini), 12);
}

// ---------------------------------------------------------------------------------------------
// Job GPU sous un flock externe (queue run lancé sous flock) : pas d'interblocage
// ---------------------------------------------------------------------------------------------

/// Aide : exécute un job GPU de la file KVU_AIDE_FILE avec le verrou KVU_AIDE_VERROU.
#[test]
fn aide_executer_gpu() {
    let (Ok(dossier), Ok(verrou)) = (std::env::var("KVU_AIDE_FILE"), std::env::var("KVU_AIDE_VERROU")) else {
        return;
    };
    let mut p = pol(Machine::Inconnue);
    p.ram_min_go = 0.0;
    p.seuil_ssd_go = 0.0;
    p.verrou_gpu = PathBuf::from(verrou);
    p.attente_verrou_s = 5;
    let f = File::ouvrir(Path::new(&dossier)).unwrap();
    let mut j = sh("echo gpu-ok");
    j.gpu = true;
    let id = f.ajouter(j).unwrap();
    let pris = f.prendre_suivant(&p).unwrap().unwrap();
    let debut = Instant::now();
    let rc = f.executer(pris, &p).unwrap();
    let journal = std::fs::read_to_string(f.chemin_journal(&id)).unwrap();
    let flock_dans_la_commande = journal.contains(" flock -w ");
    println!(
        "AIDE:GPU rc={rc} ms={} flock={flock_dans_la_commande}",
        debut.elapsed().as_millis()
    );
}

#[test]
fn job_gpu_sous_flock_externe_ne_se_bloque_pas() {
    if Command::new("flock").arg("--version").output().is_err() {
        println!("SKIP : flock(1) absent");
        return;
    }
    let lancer = |sous_flock: bool| {
        let t = Temp::nouveau("gpu_ext");
        let verrou = t.chemin().join("gpu.lock");
        let exe = std::env::current_exe().unwrap();
        let args = ["--exact", "aide_executer_gpu", "--nocapture", "--test-threads=1"];
        let mut c = if sous_flock {
            let mut c = Command::new("flock");
            c.args(["-w", "20"]).arg(&verrou).arg(&exe);
            c
        } else {
            Command::new(&exe)
        };
        let debut = Instant::now();
        let s = c
            .args(args)
            .env("KVU_AIDE_FILE", t.chemin().join("file"))
            .env("KVU_AIDE_VERROU", &verrou)
            .stdin(Stdio::null())
            .output()
            .unwrap();
        let sortie = String::from_utf8_lossy(&s.stdout).to_string();
        let ligne = sortie
            .lines()
            .find_map(|l| l.find("AIDE:GPU").map(|i| l[i..].to_string()))
            .unwrap_or_default();
        (s.status.success(), ligne, debut.elapsed())
    };
    // Sous flock externe : le job ne remet PAS de flock (ancêtre détecté) et rend vite.
    let (ok, ligne, duree) = lancer(true);
    assert!(ok && ligne.contains("rc=0") && ligne.contains("flock=false"), "{ligne}");
    assert!(duree < Duration::from_secs(3), "{duree:?}");
    // Sans flock externe : la commande est enveloppée dans flock -w.
    let (ok, ligne, _) = lancer(false);
    assert!(ok && ligne.contains("rc=0") && ligne.contains("flock=true"), "{ligne}");
}

// ---------------------------------------------------------------------------------------------
// ssh distant (faux ssh)
// ---------------------------------------------------------------------------------------------

fn faux_ssh(t: &Temp, corps: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let p = t.chemin().join("faux_ssh.sh");
    std::fs::write(&p, format!("#!/bin/sh\n[ \"${{1:-}}\" = --sonde ] && exit 0\n{corps}\n")).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    // « Text file busy » (ETXTBSY) : un autre thread qui fork pendant l'écriture garde brièvement
    // le descripteur ouvert. On sonde le script jusqu'à ce qu'il soit exécutable.
    for _ in 0..200 {
        match Command::new(&p).arg("--sonde").output() {
            Err(e) if e.raw_os_error() == Some(26) => std::thread::sleep(Duration::from_millis(5)),
            _ => break,
        }
    }
    p
}

fn job_distant() -> Job {
    let mut j = Job::nouveau(Machine::Gamer, v(&["kevinotron", "su2", "gen", "--note", "it's"]));
    j.gpu = true;
    j.mem_max = Some("8G".into());
    j.fils = 4;
    j.priorite = 2;
    j.sortie_estimee_go = 1.5;
    j.repertoire = PathBuf::from("/home/k/ym 4");
    j
}

#[test]
fn echappement_shell() {
    assert_eq!(echapper_shell("abc-1.2/x=y"), "abc-1.2/x=y");
    assert_eq!(echapper_shell(""), "''");
    assert_eq!(echapper_shell("a b"), "'a b'");
    assert_eq!(echapper_shell("it's"), "'it'\\''s'");
    assert_eq!(echapper_shell("$(rm -rf /)"), "'$(rm -rf /)'");
    assert_eq!(echapper_shell("a;b"), "'a;b'");
}

#[test]
fn commande_ssh_golden() {
    let p = pol(Machine::Acemagic);
    let argv = commande_ssh(&job_distant(), &p, "~/kevinotron_dev/bin/kevinotron").unwrap();
    assert_eq!(&argv[..4], &v(&["ssh", "-o", "BatchMode=yes", "gamer-eth"])[..]);
    assert_eq!(
        argv[4],
        "~/kevinotron_dev/bin/kevinotron queue add --machine gamer --gpu --mem 8G --fils 4 --priorite 2 --sortie-go 1.5 \
         --repertoire '/home/k/ym 4' -- kevinotron su2 gen --note 'it'\\''s'"
    );
    // Sans répertoire, sans gpu, sans mem : options omises ; binaire à chemin absolu échappé.
    let mut j = Job::nouveau(Machine::Gamer, v(&["ls"]));
    j.priorite = -3;
    let a = commande_ssh(&j, &p, "/opt/mon bin/kevin").unwrap();
    assert_eq!(
        a[4],
        "'/opt/mon bin/kevin' queue add --machine gamer --priorite -3 --sortie-go 0 -- ls"
    );
    // Pas de câble dans la politique : erreur d'usage.
    let e = commande_ssh(&j, &pol(Machine::Gamer), "x").unwrap_err();
    assert_eq!(e.code_sortie(), 2);
}

#[test]
fn ajouter_distant_avec_faux_ssh() {
    let t = Temp::nouveau("ssh_ok");
    let journal = t.chemin().join("appel.txt");
    let ssh = faux_ssh(
        &t,
        &format!(
            "for a in \"$@\"; do printf '%s\\n' \"$a\" >> {}; done\necho 'bruit'\necho 20261004-101010-gamer-0007",
            journal.display()
        ),
    );
    let id = ajouter_distant_avec(
        &job_distant(),
        &pol(Machine::Acemagic),
        ssh.to_str().unwrap(),
        "~/kevinotron_dev/bin/kevinotron",
    )
    .unwrap();
    assert_eq!(id, "20261004-101010-gamer-0007");
    let appel: Vec<String> = std::fs::read_to_string(&journal)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(&appel[..3], &v(&["-o", "BatchMode=yes", "gamer-eth"])[..]);
    assert!(
        appel[3].starts_with("~/kevinotron_dev/bin/kevinotron queue add --machine gamer"),
        "{}",
        appel[3]
    );
}

#[test]
fn ajouter_distant_echecs() {
    let t = Temp::nouveau("ssh_ko");
    let p = pol(Machine::Acemagic);
    // ssh en échec (255) : erreur d'E/S qui porte le message distant.
    let ssh = faux_ssh(&t, "echo 'Connection refused' >&2\nexit 255");
    let e = ajouter_distant_avec(&job_distant(), &p, ssh.to_str().unwrap(), "k").unwrap_err();
    assert_eq!(e.code_sortie(), 4);
    assert!(e.to_string().contains("Connection refused"), "{e}");
    // Réponse vide : erreur de format.
    let vide = faux_ssh(&t, "exit 0");
    assert!(matches!(
        ajouter_distant_avec(&job_distant(), &p, vide.to_str().unwrap(), "k"),
        Err(Erreur::Format(_))
    ));
    // Programme introuvable.
    assert_eq!(
        ajouter_distant_avec(&job_distant(), &p, "/nexiste/pas/ssh", "k")
            .unwrap_err()
            .code_sortie(),
        4
    );
}

// ---------------------------------------------------------------------------------------------
// queue (CLI) : les variables d'environnement sont globales, donc on sérialise
// ---------------------------------------------------------------------------------------------

static ENV: Mutex<()> = Mutex::new(());

fn avec_env<T>(t: &Temp, f: impl FnOnce() -> T) -> T {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("KEVINOTRON_FILE", t.chemin().join("file"));
    std::env::set_var("KEVINOTRON_MACHINE", "inconnue");
    std::env::set_var("KEVINOTRON_SSD_MIN_GO", "0");
    // hermétique : la RAM réelle de la machine (gamer chargé : 5 Go < 8 Go) ne doit pas bloquer le job de test
    std::env::set_var("KEVINOTRON_RAM_MIN_GO", "0");
    let r = f();
    for k in ["KEVINOTRON_FILE", "KEVINOTRON_MACHINE", "KEVINOTRON_SSD_MIN_GO", "KEVINOTRON_RAM_MIN_GO"] {
        std::env::remove_var(k);
    }
    r
}

fn cli(args: &[&str]) -> i32 {
    principal_machine(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
}

#[test]
fn cli_queue_add_list_run_annuler() {
    let t = Temp::nouveau("cli");
    avec_env(&t, || {
        assert_eq!(cli(&["queue", "list"]), 0, "file vide");
        let trace = t.chemin().join("trace.txt");
        let ecrire = format!("echo cli-ok >> {}", trace.display());
        assert_eq!(
            cli(&[
                "queue",
                "add",
                "--priorite",
                "3",
                "--sortie-go",
                "0.5",
                "--fils",
                "2",
                "--",
                "sh",
                "-c",
                &ecrire
            ]),
            0
        );
        assert_eq!(cli(&["queue", "add", "--", "sh", "-c", "exit 4"]), 0);
        let f = File::ouvrir(&t.chemin().join("file")).unwrap();
        let l = f.lister_etat(EtatJob::Attente).unwrap();
        assert_eq!(l.len(), 2);
        let haut = l.iter().find(|j| j.priorite == 3).unwrap();
        assert!(l.iter().any(|j| j.priorite == 0));
        assert_eq!((haut.fils, haut.sortie_estimee_go), (2, 0.5));
        assert!(haut.repertoire.is_dir(), "répertoire courant par défaut");
        // run --une : prend la plus prioritaire.
        assert_eq!(cli(&["queue", "run", "--une"]), 0);
        assert_eq!(std::fs::read_to_string(&trace).unwrap(), "cli-ok\n");
        assert_eq!(f.lister_etat(EtatJob::Fini).unwrap()[0].id, haut.id);
        // Le suivant échoue : rc 1 (ECHEC) et rangé dans echec/.
        assert_eq!(cli(&["queue", "run", "--une"]), 1);
        assert_eq!(f.lister_etat(EtatJob::Echec).unwrap()[0].rc, Some(4));
        assert_eq!(cli(&["queue", "run", "--une"]), 0, "file vide : rien à faire");
        // annuler
        assert_eq!(cli(&["queue", "add", "--", "true"]), 0);
        let id = f.lister_etat(EtatJob::Attente).unwrap()[0].id.clone();
        assert_eq!(cli(&["queue", "annuler", &id]), 0);
        assert_eq!(cli(&["queue", "annuler", &id]), 2, "déjà annulé : erreur d'usage");
        assert_eq!(cli(&["queue", "list", "--tout", "--json"]), 0);
        // vider
        assert_eq!(cli(&["queue", "add", "--", "true"]), 0);
        assert_eq!(cli(&["queue", "run", "--boucle", "--vider", "--pause-s", "1"]), 0);
        assert_eq!(f.compter(EtatJob::Attente), 0);
    });
}

#[test]
fn cli_erreurs_dusage() {
    let t = Temp::nouveau("cli_err");
    avec_env(&t, || {
        assert_eq!(cli(&["queue", "add"]), 2, "commande manquante");
        assert_eq!(cli(&["queue", "add", "--machine", "pc-de-mamie", "--", "true"]), 2);
        assert_eq!(cli(&["queue", "add", "--repertoire", "/nexiste/pas", "--", "true"]), 2);
        assert_eq!(cli(&["queue", "add", "--sortie-go", "-1", "--", "true"]), 2);
        assert_eq!(cli(&["queue"]), 2);
        assert_eq!(cli(&["inconnu"]), 2);
        assert_eq!(cli(&["etat", "--json"]), 0);
    });
}

#[test]
fn cli_queue_run_refuse_si_ssd_sous_le_seuil() {
    let t = Temp::nouveau("cli_ssd");
    avec_env(&t, || {
        std::env::set_var("KEVINOTRON_SSD_MIN_GO", "1000000000");
        assert_eq!(cli(&["queue", "add", "--", "true"]), 0);
        assert_eq!(
            cli(&["queue", "run", "--une"]),
            0,
            "rien de pris, ce n'est pas une erreur"
        );
        let f = File::ouvrir(&t.chemin().join("file")).unwrap();
        assert_eq!(f.compter(EtatJob::Attente), 1);
        assert_eq!(f.compter(EtatJob::Fini), 0);
    });
}

// ---------------------------------------------------------------------------------------------
// Jobs orphelins (vérificateur du 04/10 : un worker tué par SIGKILL laissait son job dans en_cours/)
// ---------------------------------------------------------------------------------------------

#[test]
fn stat_debut_et_zombie() {
    let stat = "1234 (nom (bizarre) x) S 1 1234 1234 0 -1 4194560 100 0 0 0 5 3 0 0 20 0 1 0 987654 1000 10";
    assert_eq!(debut_depuis_stat(stat), Some(987654));
    let zombie = stat.replace(") S ", ") Z ");
    assert_eq!(debut_depuis_stat(&zombie), None, "un zombie ne fera plus rien");
    assert_eq!(debut_depuis_stat("pas de parenthèse"), None);
    // ce processus est vivant, avec son propre instant de démarrage ; un pid réutilisé (autre début) ne l'est pas
    let moi = std::process::id();
    let debut = debut_processus(moi).expect("début de ce processus");
    assert!(processus_vivant(moi, Some(debut)));
    assert!(!processus_vivant(moi, Some(debut + 1)));
    assert!(boot_id().is_some_and(|b| b.len() >= 32));
}

#[test]
fn decision_orphelin() {
    let vivant = |pids: &'static [u32]| move |p: u32, _d: Option<u64>| pids.contains(&p);
    let mut j = job(&["x"]);
    j.boot_id = Some("boot-a".into());
    j.worker_pid = Some(10);
    j.pid = Some(20);
    // worker vivant : jamais orphelin
    assert!(!est_orphelin(&j, Some("boot-a"), &vivant(&[10, 20]), 1e6));
    // worker mort, processus du job vivant : il finira (rc perdu), pas orphelin
    assert!(!est_orphelin(&j, Some("boot-a"), &vivant(&[20]), 1e6));
    // worker et processus morts : orphelin
    assert!(est_orphelin(&j, Some("boot-a"), &vivant(&[]), 1e6));
    // autre boot : orphelin même si des pid identiques existent
    assert!(est_orphelin(&j, Some("boot-b"), &vivant(&[10, 20]), 1e6));
    // ancien format (sans worker) : délai de grâce, puis selon le processus du job
    let mut ancien = job(&["x"]);
    ancien.pid = Some(20);
    assert!(!est_orphelin(&ancien, Some("boot-a"), &vivant(&[]), 5.0));
    assert!(est_orphelin(&ancien, Some("boot-a"), &vivant(&[]), GRACE_ORPHELIN_S + 1.0));
    assert!(!est_orphelin(&ancien, Some("boot-a"), &vivant(&[20]), GRACE_ORPHELIN_S + 1.0));
}

#[test]
fn orphelin_range_en_echec_par_le_worker_suivant() {
    let t = Temp::nouveau("orphelin");
    let f = ouvrir(&t);
    let p = pol_libre(&t);
    let id = f.ajouter(sh("true")).unwrap();
    // prise par un « worker » puis disparition simulée : worker et processus inexistants (pid 0 n'est jamais un
    // processus utilisateur ; boot courant)
    let mut pris = f.prendre_suivant(&p).unwrap().unwrap();
    assert_eq!(pris.worker_pid, Some(std::process::id()));
    pris.worker_pid = Some(u32::MAX - 1);
    pris.pid = Some(u32::MAX - 2);
    std::fs::write(
        t.chemin().join(format!("file/en_cours/{id}.json")),
        serde_json::to_vec(&pris).unwrap(),
    )
    .unwrap();
    assert_eq!(f.orphelins().len(), 1);
    // le worker suivant le range en échec avant de chercher du travail ; rien n'est relancé
    assert!(f.prendre_suivant(&p).unwrap().is_none());
    assert_eq!(f.compter(EtatJob::EnCours), 0);
    let echecs = f.lister_etat(EtatJob::Echec).unwrap();
    assert_eq!(echecs.len(), 1);
    assert!(echecs[0].orphelin && echecs[0].rc.is_none());
    assert!(std::fs::read_to_string(f.chemin_journal(&id)).unwrap().contains("orphelin"));
    assert!(f.recuperer_orphelins().unwrap().is_empty(), "idempotent");
}
