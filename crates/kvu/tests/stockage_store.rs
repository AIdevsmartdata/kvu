//! Gate SSD-2 (ARCHITECTURE_UNIFIEE.md §6.5) et tests du store de la tâche T8.
//!
//! `store gc` simulé ne supprime RIEN ; `--appliquer` n'agit que sous la racine, respecte `.garder`, ne suit
//! aucun lien symbolique ; c16 → q64 vérifié par l'aller-retour. Plus : du (tri décroissant, chemins externes
//! en lecture), vérifier (sha256, journal d'éclaircissement), lister, exporter (q64 → c16 en flux).
//!
//! JAMAIS de gc sur des données réelles : tout est fabriqué sous le scratch de test.
//! Chaque test imprime `PASS <nom>`.

#[path = "donnees/stockage/support.rs"]
mod support;

use kvu::commun::{code, ChampHote, Erreur};
use kvu::heritage::manifest::nom_hote;
use kvu::stockage::garde::GardeSsd;
use kvu::stockage::npy;
use kvu::stockage::store::{PolitiqueEclaircissement, RapportGc, Store};
use kvu::stockage::{FormatChamp, Manifeste};
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::process::Command;
use support::*;

const L: usize = 2;
const JOURS3: u64 = 3 * 86_400;

fn libre() -> GardeSsd {
    GardeSsd::nouvelle(0.0)
}

fn politique(k: u64, q64: bool, zstd: bool, reprises: bool) -> PolitiqueEclaircissement {
    PolitiqueEclaircissement {
        garder_chaque: k,
        convertir_q64: q64,
        compresser_zstd: zstd,
        purger_reprises: reprises,
    }
}

fn pid_mort() -> u64 {
    let mut c = Command::new("true").spawn().unwrap();
    let pid = c.id();
    c.wait().unwrap();
    u64::from(pid)
}

fn champ(i: usize) -> ChampHote {
    champ_aleatoire(L, 1000 + i as u64)
}

/// Run COMPLETE de `n` dumps `cfg_{250 i}.npy` (c16 ou q64) + reprise, avec manifeste. Rend son répertoire.
fn run_complete(store: &Path, nom: &str, n: usize, format: FormatChamp, reprise: bool) -> PathBuf {
    let dir = store.join(nom);
    let mut m = Manifeste::ouvrir_avec_garde(&dir, "su2", &[], &libre(), 0).unwrap();
    for i in 0..n {
        let p = dir.join(format!("cfg_{:06}.npy", 250 * i));
        npy::ecrire_champ(&p, &champ(i), format, &libre()).unwrap();
        m.fichier(&p).unwrap();
    }
    fs::write(dir.join("measurements.csv"), "measurement_index,total_steps\n0,0\n").unwrap();
    m.fichier(&dir.join("measurements.csv")).unwrap();
    if reprise {
        fs::write(dir.join("reprise.json"), "{\"sweep\": 1}").unwrap();
        fs::write(dir.join("reprise.champs"), vec![7u8; 4096]).unwrap();
    }
    m.terminer(0).unwrap();
    dir
}

/// Run sans COMPLETE, pid donné, vieux de 3 jours.
fn run_sans_complete(store: &Path, nom: &str, pid: u64, age_s: u64) -> PathBuf {
    let dir = store.join(nom);
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("manifest.json"),
        format!("{{\n  \"schema\": \"kevinotron-provenance/1\",\n  \"pid\": {pid},\n  \"hote\": \"{}\",\n  \"statut\": \"en_cours\"\n}}\n", nom_nettoye(&nom_hote())),
    )
    .unwrap();
    npy::ecrire_champ(&dir.join("cfg_000000.npy"), &champ(0), FormatChamp::C16, &libre()).unwrap();
    fs::write(dir.join("therm.csv"), "step,plaquette\n").unwrap();
    vieillir_arbre(&dir, age_s);
    dir
}

fn nom_nettoye(h: &str) -> String {
    h.replace('"', "")
}

fn dumps(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("cfg_"))
        .collect();
    v.sort();
    v
}

fn raisons(r: &RapportGc) -> Vec<String> {
    r.actions.iter().map(|a| a.2.clone()).collect()
}

/// Store de test complet : run COMPLETE (10 dumps c16 + reprise), orphelin vieux (pid mort), run vivant, run mort
/// mais récent, run protégé par .garder, lien vers un répertoire EXTERNE contenant un orphelin.
struct Banc {
    sc: Scratch,
    store: PathBuf,
    externe: PathBuf,
}

fn banc(nom: &str) -> Banc {
    let sc = Scratch::nouveau(nom);
    let store = sc.p("store");
    fs::create_dir_all(&store).unwrap();
    run_complete(&store, "run_ok", 10, FormatChamp::C16, true);
    run_sans_complete(&store, "orphelin", pid_mort(), JOURS3);
    run_sans_complete(&store, "vivant", u64::from(std::process::id()), JOURS3);
    run_sans_complete(&store, "recent_mort", pid_mort(), 60);
    let garde = run_sans_complete(&store, "garde", pid_mort(), JOURS3);
    fs::write(garde.join(".garder"), "").unwrap();
    // externe : un orphelin que gc ne doit jamais atteindre, même par un lien placé dans le store
    let externe = sc.p("externe");
    fs::create_dir_all(&externe).unwrap();
    run_sans_complete(&externe, "orphelin_externe", pid_mort(), JOURS3);
    run_complete(&externe, "run_externe", 6, FormatChamp::C16, true);
    symlink(externe.join("orphelin_externe"), store.join("lien_vers_orphelin")).unwrap();
    symlink(&externe, store.join("lien_vers_externe")).unwrap();
    Banc { sc, store, externe }
}

// ------------------------------------------------------------------------------------------
// SSD-2
// ------------------------------------------------------------------------------------------

#[test]
fn ssd2_gc_simule_ne_supprime_rien() {
    let b = banc("ssd2_sim");
    let avant = empreinte_arbre(&b.sc.0);
    let s = Store::ouvrir(&b.store).unwrap();
    let r = s
        .gc_avec_garde(&politique(3, true, false, true), 24.0 * 3600.0, false, &libre())
        .unwrap();
    assert!(!r.applique);
    assert_eq!(empreinte_arbre(&b.sc.0), avant, "la simulation a modifié le disque !");
    let rs = raisons(&r);
    for attendu in ["run-orphelin", "reprise", "eclaircissement 1/3", "conversion-q64"] {
        assert!(rs.iter().any(|x| x == attendu), "action {attendu} absente de {rs:?}");
    }
    // orphelin : seul « orphelin » est prévu (pas vivant, pas récent, pas gardé, pas les liens)
    let orphelins: Vec<_> = r.actions.iter().filter(|a| a.2 == "run-orphelin").collect();
    assert_eq!(orphelins.len(), 1);
    assert!(orphelins[0].0.ends_with("store/orphelin"));
    assert!(orphelins[0].1 > 0);
    assert!(r.octets_liberes > 0);
    // 10 dumps, un sur 3 : positions 1,2,4,5,7,8 supprimées = 6 éclaircissements ; reprises : 2 fichiers
    assert_eq!(rs.iter().filter(|x| x.starts_with("eclaircissement")).count(), 6);
    assert_eq!(rs.iter().filter(|x| *x == "reprise").count(), 2);
    assert_eq!(rs.iter().filter(|x| *x == "conversion-q64").count(), 4);
    // ce qu'on a laissé en l'état est dit
    let ign: Vec<String> = r.ignores.iter().map(|i| format!("{}|{}", i.0.display(), i.1)).collect();
    assert!(
        ign.iter()
            .any(|i| i.contains("store/vivant") && i.contains("vit encore")),
        "{ign:?}"
    );
    assert!(
        ign.iter()
            .any(|i| i.contains("store/recent_mort") && i.contains("trop récent")),
        "{ign:?}"
    );
    assert!(
        ign.iter().any(|i| i.contains("store/garde") && i.contains(".garder")),
        "{ign:?}"
    );
    println!(
        "PASS SSD-2 gc simulé : rien supprimé ({} actions prévues, {} octets libérables)",
        r.actions.len(),
        r.octets_liberes
    );
}

#[test]
fn ssd2_appliquer_sous_la_racine_garder_et_liens() {
    let b = banc("ssd2_app");
    let s = Store::ouvrir(&b.store).unwrap();
    let externe_avant = empreinte_arbre(&b.externe);
    let garde_avant = empreinte_arbre(&b.store.join("garde"));
    let vivant_avant = empreinte_arbre(&b.store.join("vivant"));
    let recent_avant = empreinte_arbre(&b.store.join("recent_mort"));
    let manifeste_avant = sha256sum(&b.store.join("run_ok/manifest.json"));
    let complete_avant = sha256sum(&b.store.join("run_ok/COMPLETE"));
    let champs: Vec<ChampHote> = (0..10)
        .map(|i| npy::lire_champ(&b.store.join(format!("run_ok/cfg_{:06}.npy", 250 * i))).unwrap())
        .collect();

    let r = s
        .gc_avec_garde(&politique(3, true, false, true), 24.0 * 3600.0, true, &libre())
        .unwrap();
    assert!(r.applique);

    // l'orphelin a disparu ; vivant, récent, protégé : intacts
    assert!(!b.store.join("orphelin").exists());
    assert_eq!(
        empreinte_arbre(&b.store.join("garde")),
        garde_avant,
        ".garder non respecté"
    );
    assert_eq!(empreinte_arbre(&b.store.join("vivant")), vivant_avant);
    assert_eq!(empreinte_arbre(&b.store.join("recent_mort")), recent_avant);
    // les liens symboliques n'ont pas été suivis ni supprimés ; l'extérieur est strictement intact
    assert_eq!(
        empreinte_arbre(&b.externe),
        externe_avant,
        "gc a touché au répertoire externe !"
    );
    assert!(fs::symlink_metadata(b.store.join("lien_vers_orphelin"))
        .unwrap()
        .file_type()
        .is_symlink());
    assert!(fs::symlink_metadata(b.store.join("lien_vers_externe"))
        .unwrap()
        .file_type()
        .is_symlink());

    // run_ok : reprises purgées, un dump sur 3 (positions 0, 3, 6, 9 = indices 0, 750, 1500, 2250)
    assert!(!b.store.join("run_ok/reprise.json").exists() && !b.store.join("run_ok/reprise.champs").exists());
    assert_eq!(
        dumps(&b.store.join("run_ok")),
        ["cfg_000000.npy", "cfg_000750.npy", "cfg_001500.npy", "cfg_002250.npy"]
    );
    // les dumps gardés ont été convertis en q64, exactement
    for (pos, i) in [0usize, 3, 6, 9].into_iter().enumerate() {
        let p = b.store.join(format!("run_ok/cfg_{:06}.npy", 250 * i));
        assert_eq!(npy::inspecter_champ(&p).unwrap(), (FormatChamp::Q64, L), "dump {pos}");
        assert_eq!(
            npy::lire_champ(&p).unwrap(),
            champs[i],
            "dump {i} : q64 ≠ c16 d'origine"
        );
        assert_eq!(fs::metadata(&p).unwrap().len(), npy::octets_champ(L, FormatChamp::Q64));
    }
    // manifest.json et COMPLETE ne sont jamais réécrits ; le journal rend compte de tout
    assert_eq!(sha256sum(&b.store.join("run_ok/manifest.json")), manifeste_avant);
    assert_eq!(sha256sum(&b.store.join("run_ok/COMPLETE")), complete_avant);
    let journal = fs::read_to_string(b.store.join("run_ok/ECLAIRCISSEMENT.jsonl")).unwrap();
    assert_eq!(journal.matches("\"action\":\"supprime\"").count(), 6 + 2);
    assert_eq!(journal.matches("\"action\":\"convertit\"").count(), 4);
    // et la vérification des sha256 reste verte, en tenant compte du journal
    s.verifier("run_ok").unwrap();
    let rv = s.verifier_rapport("run_ok").unwrap();
    assert!(rv.ok());
    assert_eq!(
        rv.fichiers_verifies, 5,
        "4 dumps convertis + measurements.csv ; les 6 supprimés sont journalisés"
    );

    // idempotent : un second gc n'a plus rien à faire de destructeur
    let r2 = s
        .gc_avec_garde(&politique(3, true, false, true), 24.0 * 3600.0, true, &libre())
        .unwrap();
    assert!(r2.actions.is_empty(), "second gc : {:?}", r2.actions);
    // l'altération d'un dump gardé est détectée
    let p = b.store.join("run_ok/cfg_000000.npy");
    let mut bytes = fs::read(&p).unwrap();
    let n = bytes.len();
    bytes[n - 5] ^= 1;
    fs::write(&p, bytes).unwrap();
    assert!(s.verifier("run_ok").is_err());
    println!("PASS SSD-2 --appliquer : sous la racine, .garder respecté, liens non suivis, c16 -> q64 vérifié ({} octets libérés)", r.octets_liberes);
}

#[test]
fn gc_eclaircissement_idempotent_puis_progressif() {
    let sc = Scratch::nouveau("gc_prog");
    let store = sc.p("store");
    fs::create_dir_all(&store).unwrap();
    run_complete(&store, "r", 13, FormatChamp::C16, false);
    let s = Store::ouvrir(&store).unwrap();
    // 1 sur 4 : positions 0, 4, 8, 12 (12 = dernier)
    s.gc_avec_garde(&politique(4, false, false, true), 3600.0, true, &libre())
        .unwrap();
    assert_eq!(dumps(&store.join("r")).len(), 4);
    // même k, autant de fois qu'on veut : rien de plus (un cron ne doit pas ronger le store)
    for _ in 0..3 {
        let r = s
            .gc_avec_garde(&politique(4, false, false, true), 3600.0, true, &libre())
            .unwrap();
        assert!(r.actions.is_empty(), "{:?}", r.actions);
        let r = s
            .gc_avec_garde(&politique(2, false, false, true), 3600.0, true, &libre())
            .unwrap();
        assert!(
            r.actions.is_empty(),
            "un k plus petit ne ré-éclaircit pas : {:?}",
            r.actions
        );
    }
    assert_eq!(dumps(&store.join("r")).len(), 4);
    // un k strictement plus grand éclaircit ce qui reste : positions 0 et 3 (premier et dernier)
    s.gc_avec_garde(&politique(8, false, false, true), 3600.0, true, &libre())
        .unwrap();
    assert_eq!(dumps(&store.join("r")), ["cfg_000000.npy", "cfg_003000.npy"]);
    s.verifier("r").unwrap();
    println!("PASS gc : éclaircissement idempotent (même k), progressif (k plus grand)");
}

#[test]
fn ssd2_garder_dans_un_run_complete_et_a_la_racine() {
    let sc = Scratch::nouveau("ssd2_garder");
    let store = sc.p("store");
    fs::create_dir_all(&store).unwrap();
    run_complete(&store, "protege", 10, FormatChamp::C16, true);
    run_complete(&store, "libre", 10, FormatChamp::C16, true);
    fs::write(store.join("protege/.garder"), "").unwrap();
    let protege_avant = empreinte_arbre(&store.join("protege"));
    let s = Store::ouvrir(&store).unwrap();
    let r = s
        .gc_avec_garde(&politique(2, true, false, true), 3600.0, true, &libre())
        .unwrap();
    assert_eq!(
        empreinte_arbre(&store.join("protege")),
        protege_avant,
        ".garder dans un run COMPLETE non respecté"
    );
    assert!(
        r.actions.iter().all(|a| !a.0.starts_with(store.join("protege"))),
        "{:?}",
        r.actions
    );
    assert!(
        !store.join("libre/reprise.json").exists(),
        "le run non protégé a bien été éclairci"
    );
    // un .garder dans un sous-répertoire ne protège que ce sous-arbre
    run_complete(&store, "partiel", 4, FormatChamp::C16, true);
    fs::create_dir_all(store.join("partiel/chaine_a")).unwrap();
    fs::create_dir_all(store.join("partiel/chaine_b")).unwrap();
    for i in 0..5 {
        for c in ["chaine_a", "chaine_b"] {
            npy::ecrire_champ(
                &store.join(format!("partiel/{c}/cfg_{:06}.npy", i)),
                &champ(i),
                FormatChamp::C16,
                &libre(),
            )
            .unwrap();
        }
    }
    fs::write(store.join("partiel/chaine_b/.garder"), "").unwrap();
    s.gc_avec_garde(&politique(2, false, false, false), 3600.0, true, &libre())
        .unwrap();
    assert_eq!(dumps(&store.join("partiel/chaine_b")).len(), 5, "chaine_b est protégée");
    assert_eq!(
        dumps(&store.join("partiel/chaine_a")),
        ["cfg_000000.npy", "cfg_000002.npy", "cfg_000004.npy"],
        "chaine_a : 1 sur 2, dernier gardé"
    );
    // .garder à la racine : tout le store est protégé
    fs::write(store.join(".garder"), "").unwrap();
    let avant = empreinte_arbre(&store);
    let r = s
        .gc_avec_garde(&politique(2, true, true, true), 0.0, true, &libre())
        .unwrap();
    assert!(r.actions.is_empty());
    assert_eq!(empreinte_arbre(&store), avant);
    println!("PASS SSD-2 .garder respecté (run, sous-répertoire, racine)");
}

#[test]
fn ssd2_liens_dans_un_orphelin_et_boucles() {
    let sc = Scratch::nouveau("ssd2_liens");
    let store = sc.p("store");
    let dehors = sc.p("dehors");
    fs::create_dir_all(&store).unwrap();
    fs::create_dir_all(&dehors).unwrap();
    fs::write(dehors.join("precieux.npy"), "données précieuses").unwrap();
    // orphelin qui contient un lien vers l'extérieur et une boucle
    let o = run_sans_complete(&store, "orphelin", pid_mort(), JOURS3);
    symlink(&dehors, o.join("lien_dehors")).unwrap();
    symlink(&o, o.join("boucle")).unwrap();
    symlink("n_existe_pas", o.join("pendouillant")).unwrap();
    // et des liens à la racine du store : boucle, pendouillant
    symlink(&store, store.join("boucle_racine")).unwrap();
    symlink("rien", store.join("pendouillant")).unwrap();
    let avant = empreinte_arbre(&dehors);
    let s = Store::ouvrir(&store).unwrap();
    // du ne suit pas non plus les liens
    let du = s.du(&[]).unwrap();
    assert!(du
        .iter()
        .all(|l| !l.chemin.ends_with("boucle_racine") && !l.chemin.ends_with("pendouillant")));
    let r = s
        .gc_avec_garde(&politique(1, false, false, true), 3600.0, true, &libre())
        .unwrap();
    assert_eq!(r.actions.len(), 1);
    assert!(!store.join("orphelin").exists());
    assert_eq!(empreinte_arbre(&dehors), avant, "la cible du lien a été touchée");
    assert!(dehors.join("precieux.npy").exists());
    println!("PASS SSD-2 liens symboliques (extérieur, boucle, pendouillant) jamais suivis");
}

#[test]
fn ssd2_gc_refuse_les_racines_absurdes() {
    for racine in [
        "/".to_string(),
        std::env::var("HOME").unwrap_or_else(|_| ".".into()),
    ] {
        let s = Store {
            racine: PathBuf::from(&racine),
        };
        let e = s
            .gc_avec_garde(&politique(2, true, true, true), 0.0, false, &libre())
            .unwrap_err();
        assert!(matches!(e, Erreur::Config(_)), "{racine} : {e}");
    }
    assert!(Store::ouvrir(Path::new("/chemin/inexistant/kvu_t8")).is_err());
    println!("PASS SSD-2 gc refuse « / » et le répertoire personnel comme racine");
}

#[test]
fn gc_fichiers_partiels_abandonnes_d_un_run_complete() {
    let sc = Scratch::nouveau("gc_part");
    let store = sc.p("store");
    fs::create_dir_all(&store).unwrap();
    let dir = run_complete(&store, "r", 2, FormatChamp::C16, false);
    fs::write(dir.join("cfg_000500.npy.part"), vec![0u8; 1000]).unwrap();
    fs::write(dir.join("recent.npy.part"), vec![0u8; 10]).unwrap();
    vieillir(&dir.join("cfg_000500.npy.part"), JOURS3);
    let s = Store::ouvrir(&store).unwrap();
    let r = s
        .gc_avec_garde(&PolitiqueEclaircissement::prudente(), 24.0 * 3600.0, true, &libre())
        .unwrap();
    assert_eq!(raisons(&r), ["partiel-orphelin"]);
    assert!(!dir.join("cfg_000500.npy.part").exists());
    assert!(
        dir.join("recent.npy.part").exists(),
        "un .part récent peut être en cours d'écriture"
    );
    println!("PASS gc : .part abandonné supprimé, .part récent conservé");
}

#[test]
fn ssd2_seuil_age_des_orphelins() {
    let sc = Scratch::nouveau("ssd2_age");
    let store = sc.p("store");
    fs::create_dir_all(&store).unwrap();
    run_sans_complete(&store, "vieux", pid_mort(), 5 * 3600);
    let s = Store::ouvrir(&store).unwrap();
    let p = PolitiqueEclaircissement::prudente();
    // 5 h < 24 h : on garde
    assert!(s
        .gc_avec_garde(&p, 24.0 * 3600.0, true, &libre())
        .unwrap()
        .actions
        .is_empty());
    assert!(store.join("vieux").exists());
    // seuil ramené à 1 h : supprimé
    assert_eq!(s.gc_avec_garde(&p, 3600.0, true, &libre()).unwrap().actions.len(), 1);
    assert!(!store.join("vieux").exists());
    println!("PASS SSD-2 âge minimal des orphelins respecté (24 h par défaut)");
}

#[test]
fn ssd2_hote_ou_pid_inconnu_laisse_en_place() {
    let sc = Scratch::nouveau("ssd2_hote");
    let store = sc.p("store");
    fs::create_dir_all(store.join("autre_hote")).unwrap();
    fs::write(
        store.join("autre_hote/manifest.json"),
        format!("{{\"pid\": {}, \"hote\": \"une-autre-machine\"}}", pid_mort()),
    )
    .unwrap();
    fs::write(store.join("autre_hote/cfg_000000.npy"), "x").unwrap();
    fs::create_dir_all(store.join("sans_pid")).unwrap();
    fs::write(store.join("sans_pid/manifest.json"), "{ illisible").unwrap();
    vieillir_arbre(&store, JOURS3);
    let s = Store::ouvrir(&store).unwrap();
    let r = s
        .gc_avec_garde(&politique(1, false, false, true), 3600.0, true, &libre())
        .unwrap();
    assert!(
        r.actions.is_empty(),
        "un pid d'une autre machine ne prouve rien : {:?}",
        r.actions
    );
    assert!(store.join("autre_hote").exists() && store.join("sans_pid").exists());
    println!("PASS SSD-2 run d'un autre hôte ou au manifeste illisible : laissé en place");
}

// ------------------------------------------------------------------------------------------
// du, lister, vérifier
// ------------------------------------------------------------------------------------------

#[test]
fn du_trie_decroissant_et_externe_en_lecture() {
    let b = banc("du");
    let s = Store::ouvrir(&b.store).unwrap();
    let ext = b.sc.p("ext_du");
    fs::create_dir_all(&ext).unwrap();
    fs::write(ext.join("gros.bin"), vec![1u8; 3_000_000]).unwrap();
    let ext_avant = empreinte_arbre(&ext);
    let lignes = s.du(std::slice::from_ref(&ext)).unwrap();
    assert!(lignes.windows(2).all(|w| w[0].octets >= w[1].octets), "tri décroissant");
    assert_eq!(lignes[0].chemin, ext, "le plus gros d'abord");
    assert_eq!(lignes[0].octets, 3_000_000);
    let ok = lignes.iter().find(|l| l.chemin.ends_with("run_ok")).unwrap();
    assert!(ok.complete && ok.fichiers >= 12);
    let sans = lignes.iter().find(|l| l.chemin.ends_with("orphelin")).unwrap();
    assert!(!sans.complete && sans.age_s > 2.9 * 86_400.0);
    // les liens du store ne comptent pas, l'externe est intact
    assert!(lignes.iter().all(|l| !l.chemin.ends_with("lien_vers_externe")));
    assert_eq!(empreinte_arbre(&ext), ext_avant);
    println!("PASS du trié décroissant, externes en lecture seule, liens ignorés");
}

#[test]
fn lister_rend_les_champs_du_manifeste() {
    let b = banc("lister");
    let s = Store::ouvrir(&b.store).unwrap();
    let l = s.lister().unwrap();
    let ok = l.iter().find(|e| e.id == "run_ok").unwrap();
    assert!(ok.complete && !ok.garder);
    assert_eq!((ok.moteur.as_deref(), ok.rc), (Some("su2"), Some(0)));
    assert!(ok.machine.is_some() && ok.debut_unix.unwrap() > 1.7e9);
    assert!(l.iter().find(|e| e.id == "garde").unwrap().garder);
    assert!(!l.iter().find(|e| e.id == "orphelin").unwrap().complete);
    assert!(l.windows(2).all(|w| w[0].id <= w[1].id));
    println!("PASS lister");
}

#[test]
fn verifier_detecte_octet_altere_fichier_manquant_et_complete_fausse() {
    let sc = Scratch::nouveau("verifier");
    let store = sc.p("store");
    fs::create_dir_all(&store).unwrap();
    let dir = run_complete(&store, "r", 3, FormatChamp::C16, false);
    let s = Store::ouvrir(&store).unwrap();
    s.verifier("r").unwrap();

    // un octet altéré dans un dump
    let p = dir.join("cfg_000250.npy");
    let orig = fs::read(&p).unwrap();
    let mut mauvais = orig.clone();
    mauvais[300] ^= 0x10;
    fs::write(&p, &mauvais).unwrap();
    let e = s.verifier("r").unwrap_err();
    assert_eq!(e.code_sortie(), code::VALIDATION);
    assert!(e.to_string().contains("sha256"), "{e}");
    fs::write(&p, &orig).unwrap();
    s.verifier("r").unwrap();

    // fichier manquant et non journalisé
    fs::remove_file(&p).unwrap();
    let rv = s.verifier_rapport("r").unwrap();
    assert_eq!(rv.ecarts.len(), 1);
    assert_eq!(rv.ecarts[0].nature, "manquant");
    fs::write(&p, &orig).unwrap();

    // manifest.json modifié : le lien COMPLETE ne tient plus
    let m = fs::read_to_string(dir.join("manifest.json")).unwrap();
    fs::write(
        dir.join("manifest.json"),
        m.replace("\"moteur\": \"su2\"", "\"moteur\": \"su3\""),
    )
    .unwrap();
    let rv = s.verifier_rapport("r").unwrap();
    assert!(rv.ecarts.iter().any(|e| e.nature == "complete"), "{:?}", rv.ecarts);
    fs::write(dir.join("manifest.json"), m).unwrap();
    s.verifier("r").unwrap();

    // COMPLETE absent
    let c = fs::read_to_string(dir.join("COMPLETE")).unwrap();
    fs::remove_file(dir.join("COMPLETE")).unwrap();
    assert!(s.verifier("r").is_err());
    fs::write(dir.join("COMPLETE"), c).unwrap();

    // identifiants invalides
    for id in ["../x", "/etc", "", "a/../../b", "inexistant"] {
        assert!(
            matches!(s.verifier(id), Err(Erreur::Config(_))),
            "« {id} » aurait dû être refusé"
        );
    }
    println!("PASS verifier : octet altéré, fichier manquant, COMPLETE faux ou absent, identifiants invalides");
}

#[test]
fn verifier_suit_les_runs_imbriques() {
    let sc = Scratch::nouveau("verifier_nid");
    let store = sc.p("store");
    fs::create_dir_all(store.join("ens")).unwrap();
    run_complete(&store.join("ens"), "c000", 2, FormatChamp::C16, false);
    run_complete(&store.join("ens"), "c001", 2, FormatChamp::Q64, false);
    let s = Store::ouvrir(&store).unwrap();
    let rv = s.verifier_rapport("ens").unwrap();
    assert!(rv.ok());
    assert_eq!((rv.unites, rv.fichiers_verifies), (2, 6));
    // une altération dans c001 est vue
    let p = store.join("ens/c001/cfg_000000.npy");
    let mut b = fs::read(&p).unwrap();
    b[200] ^= 1;
    fs::write(&p, b).unwrap();
    assert!(s.verifier("ens").is_err());
    println!("PASS verifier suit les unités imbriquées (un run par chaîne)");
}

// ------------------------------------------------------------------------------------------
// zstd et export
// ------------------------------------------------------------------------------------------

#[test]
fn gc_compression_zstd_et_verification() {
    let sc = Scratch::nouveau("gc_zstd");
    let store = sc.p("store");
    fs::create_dir_all(&store).unwrap();
    run_complete(&store, "r", 4, FormatChamp::Q64, true);
    let s = Store::ouvrir(&store).unwrap();
    let c0 = champ(0);
    // simulation : annoncée, rien de fait
    let r = s
        .gc_avec_garde(&politique(1, false, true, false), 3600.0, false, &libre())
        .unwrap();
    assert!(raisons(&r).iter().any(|x| x.starts_with("compression-zstd")));
    assert!(store.join("r/cfg_000000.npy").exists());
    // application
    let r = s
        .gc_avec_garde(&politique(1, false, true, false), 3600.0, true, &libre())
        .unwrap();
    assert_eq!(raisons(&r).iter().filter(|x| *x == "compression-zstd").count(), 4);
    assert_eq!(
        dumps(&store.join("r")),
        [
            "cfg_000000.npy.zst",
            "cfg_000250.npy.zst",
            "cfg_000500.npy.zst",
            "cfg_000750.npy.zst"
        ]
    );
    assert_eq!(npy::lire_champ(&store.join("r/cfg_000000.npy.zst")).unwrap(), c0);
    s.verifier("r").unwrap();
    let rv = s.verifier_rapport("r").unwrap();
    assert_eq!(rv.fichiers_verifies, 5);
    assert!(
        fs::read_to_string(store.join("r/ECLAIRCISSEMENT.jsonl"))
            .unwrap()
            .matches("compresse")
            .count()
            >= 4
    );
    // un .zst corrompu est vu par verifier
    let p = store.join("r/cfg_000250.npy.zst");
    let mut b = fs::read(&p).unwrap();
    let m = b.len() / 2;
    b[m] ^= 0x55;
    fs::write(&p, b).unwrap();
    assert!(s.verifier("r").is_err());
    println!("PASS gc --zstd : dumps compressés, journal, verifier relit le contenu décompressé");
}

#[test]
fn exporter_q64_vers_c16_en_flux() {
    let sc = Scratch::nouveau("exporter");
    let store = sc.p("store");
    fs::create_dir_all(&store).unwrap();
    run_complete(&store, "r", 3, FormatChamp::Q64, true);
    let s = Store::ouvrir(&store).unwrap();
    // un dump compressé aussi
    npy::ecrire_champ(&store.join("r/cfg_000750.npy"), &champ(3), FormatChamp::Q64, &libre()).unwrap();
    kvu::stockage::compression::compresser(&store.join("r/cfg_000750.npy")).unwrap();
    let dest = sc.p("export");
    let r = s.exporter("r", &dest, FormatChamp::C16, &libre()).unwrap();
    assert_eq!(r.convertis, 4);
    // chaque dump exporté = ce qu'écrirait directement ecrire_champ en c16 (donc su2_hip), à l'octet
    for i in 0..4 {
        let attendu = sc.p(&format!("attendu_{i}.npy"));
        npy::ecrire_champ(&attendu, &champ(i), FormatChamp::C16, &libre()).unwrap();
        let exporte = dest.join(format!("cfg_{:06}.npy", 250 * i));
        assert_eq!(fs::read(&exporte).unwrap(), fs::read(&attendu).unwrap(), "dump {i}");
    }
    assert!(
        dest.join("manifest.json").exists()
            && dest.join("measurements.csv").exists()
            && dest.join("EXPORT.json").exists()
    );
    assert!(
        !dest.join("reprise.json").exists() && !dest.join("reprise.champs").exists(),
        "les reprises ne s'exportent pas"
    );
    assert!(!dest.join("cfg_000750.npy.zst").exists());
    // destination existante : refusée
    assert!(s.exporter("r", &dest, FormatChamp::C16, &libre()).is_err());
    // garde SSD : refus avant de créer quoi que ce soit
    let libre_go = GardeSsd::libre_go(&sc.0).unwrap();
    let e = s
        .exporter(
            "r",
            &sc.p("export2"),
            FormatChamp::C16,
            &GardeSsd::nouvelle(libre_go + 1.0),
        )
        .unwrap_err();
    assert_eq!(e.code_sortie(), 7);
    assert!(!sc.p("export2").exists());
    // exporter vers q64 le même ensemble : les dumps q64 sont recopiés tels quels
    let d2 = sc.p("export_q");
    let r = s.exporter("r", &d2, FormatChamp::Q64, &libre()).unwrap();
    assert_eq!(r.convertis, 0);
    assert_eq!(
        fs::read(d2.join("cfg_000250.npy")).unwrap(),
        fs::read(store.join("r/cfg_000250.npy")).unwrap()
    );
    println!("PASS exporter q64 -> c16 en flux, identique à l'octet à l'écriture directe");
}

// ------------------------------------------------------------------------------------------
// CLI de bout en bout (in-process : commandes() + executer())
// ------------------------------------------------------------------------------------------

fn lancer(args: &[&str]) -> Result<i32, Erreur> {
    let m = kvu::stockage::cli::commandes()
        .into_iter()
        .find(|c| c.get_name() == args[0])
        .expect("commande connue")
        .try_get_matches_from(args)
        .map_err(|e| Erreur::Config(e.to_string()))?;
    kvu::stockage::cli::executer(args[0], &m)
}

#[test]
fn cli_store_de_bout_en_bout() {
    let sc = Scratch::nouveau("cli");
    let store = sc.p("store");
    fs::create_dir_all(&store).unwrap();
    run_complete(&store, "r", 6, FormatChamp::C16, true);
    run_sans_complete(&store, "orphelin", pid_mort(), JOURS3);
    let st = store.to_str().unwrap().to_string();
    let avant = empreinte_arbre(&store);

    assert_eq!(
        lancer(&["store", "--store", &st, "du", "--projeter", "64:1:20:c16"]).unwrap(),
        0
    );
    assert_eq!(
        lancer(&["store", "--store", &st, "du", "--json", "--chemin", &st]).unwrap(),
        0
    );
    assert_eq!(lancer(&["store", "--store", &st, "lister"]).unwrap(), 0);
    assert_eq!(lancer(&["store", "--store", &st, "verifier", "r"]).unwrap(), 0);
    // gc sans --appliquer : simulation, rien ne bouge
    assert_eq!(
        lancer(&["store", "--store", &st, "gc", "--garder-chaque", "2", "--convertir-q64"]).unwrap(),
        0
    );
    assert_eq!(empreinte_arbre(&store), avant, "gc sans --appliquer a modifié le store");
    // avec --appliquer
    assert_eq!(
        lancer(&[
            "store",
            "--store",
            &st,
            "gc",
            "--appliquer",
            "--garder-chaque",
            "2",
            "--convertir-q64"
        ])
        .unwrap(),
        0
    );
    assert!(!store.join("orphelin").exists());
    // 6 dumps, un sur 2 : positions 0, 2, 4 et le dernier (5)
    assert_eq!(dumps(&store.join("r")).len(), 4);
    assert_eq!(lancer(&["store", "--store", &st, "verifier", "r"]).unwrap(), 0);
    // export vers c16 des dumps (devenus q64 par la conversion)
    let dest = sc.p("export");
    assert_eq!(
        lancer(&["store", "--store", &st, "exporter", "r", dest.to_str().unwrap()]).unwrap(),
        0
    );
    assert_eq!(
        npy::inspecter_champ(&dest.join("cfg_000000.npy")).unwrap().0,
        FormatChamp::C16
    );
    assert_eq!(npy::lire_champ(&dest.join("cfg_000000.npy")).unwrap(), champ(0));
    assert!(
        lancer(&["store", "--store", &st, "exporter", "r", dest.to_str().unwrap()]).is_err(),
        "destination existante"
    );

    // codes de sortie : store inexistant = usage (2), altération = validation (8), format inconnu = usage (2)
    let e = lancer(&["store", "--store", "/chemin/inexistant/kvu_t8", "du"]).unwrap_err();
    assert_eq!(e.code_sortie(), 2);
    let p = store.join("r/cfg_000000.npy");
    let mut b = fs::read(&p).unwrap();
    let n = b.len();
    b[n - 9] ^= 0x01;
    fs::write(&p, b).unwrap();
    assert_eq!(
        lancer(&["store", "--store", &st, "verifier", "r"])
            .unwrap_err()
            .code_sortie(),
        8
    );
    assert_eq!(
        lancer(&[
            "store",
            "--store",
            &st,
            "exporter",
            "r",
            sc.p("e2").to_str().unwrap(),
            "--format",
            "f32"
        ])
        .unwrap_err()
        .code_sortie(),
        2
    );
    assert!(lancer(&["store"]).is_err(), "sous-commande requise");
    println!("PASS CLI store : du, lister, verifier, gc (simulation puis --appliquer), exporter, codes de sortie");
}
