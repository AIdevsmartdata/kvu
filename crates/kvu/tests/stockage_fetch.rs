//! Gate FETCH-1 (ARCHITECTURE_UNIFIEE.md §6.5) : rapatriement par le câble 2,5 Gbit/s (gamer-eth).
//!
//! - fetch d'un petit ensemble du gamer : sha256 vérifiés ;
//! - un octet altéré à distance ⇒ erreur ET fichier fautif supprimé ;
//! - garde SSD appliquée sur la taille DISTANTE (refus avant tout octet copié, rc 7).
//!
//! Le test d'intégration `fetch1_*` est `#[ignore = "gamer"]` : il crée un petit répertoire de test sur le
//! gamer (`~/kevinotron_dev/tests_fetch/ens_<pid>`, quelques Ko plus un fichier CREUX de 3 Go qui n'occupe
//! aucun bloc), puis le supprime à la fin, même en cas d'échec. Lancement :
//!   cargo test -p kvu --test stockage_fetch -- --ignored --nocapture
//! Les tests purs (commande rsync, analyse des sources) tournent sans le gamer.

#[path = "donnees/stockage/support.rs"]
mod support;

use kvu::commun::Erreur;
use kvu::stockage::fetch::{analyser_source, commande_rsync, commande_ssh, commande_taille, recuperer};
use kvu::stockage::garde::GardeSsd;
use kvu::stockage::{npy, FormatChamp, Manifeste};
use std::fs;
use std::path::Path;
use std::process::Command;
use support::*;

const CABLE: &str = "gamer-eth";

fn libre() -> GardeSsd {
    GardeSsd::nouvelle(0.0)
}

/// Exécute une commande sur le gamer par le câble ; rend (succès, sortie standard).
fn ssh(cmd: &str) -> (bool, String) {
    let o = Command::new("ssh")
        .args(["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", CABLE, cmd])
        .output()
        .expect("lancement de ssh");
    (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned())
}

/// Supprime le répertoire de test distant à la destruction (même en cas d'échec du test).
struct NettoyageDistant(String);

impl Drop for NettoyageDistant {
    fn drop(&mut self) {
        let _ = ssh(&format!(
            "rm -rf '{}' && rmdir kevinotron_dev/tests_fetch 2>/dev/null; true",
            self.0
        ));
    }
}

#[test]
fn analyse_et_commandes_pures() {
    assert_eq!(
        analyser_source("gamer:store/run1").unwrap().chemin,
        "kevinotron_store/run1"
    );
    let c = commande_rsync("gamer:store/run1/", Path::new("/data/run1"), CABLE);
    assert_eq!(c[..5], ["rsync", "-a", "--partial", "-e", "ssh -o BatchMode=yes"]);
    assert_eq!(
        (c[5].as_str(), c[6].as_str()),
        ("gamer-eth:kevinotron_store/run1/", "/data/run1/")
    );
    let s = commande_ssh(CABLE, &commande_taille("a/b"));
    assert_eq!(s[0], "ssh");
    assert!(s.contains(&"BatchMode=yes".to_string()));
    // sources dangereuses refusées avant tout lancement
    let sc = Scratch::nouveau("fetch_pur");
    for mauvais in ["gamer:../x", "gamer:/a;reboot", "pc:/x", "gamer:$(id)", "gamer:-e"] {
        assert!(recuperer(mauvais, &sc.p("d"), &libre(), CABLE).is_err());
    }
    assert!(!sc.p("d").exists());
    println!("PASS fetch : analyse des sources et commande rsync (pures), sources dangereuses refusées");
}

#[test]
#[ignore = "gamer"]
fn fetch1_gamer_par_le_cable() {
    let (ok, hote) = ssh("hostname");
    assert!(ok, "gamer-eth injoignable par le câble : FETCH-1 ne peut pas passer");
    println!("gamer : {}", hote.trim());

    let sc = Scratch::nouveau("fetch1");
    // 1. petit ensemble local, avec son manifeste COMPLETE
    let src = sc.p("ens");
    let mut m = Manifeste::ouvrir_avec_garde(&src, "su2", &[], &libre(), 0).unwrap();
    for i in 0..3 {
        let p = src.join(format!("cfg_{:06}.npy", 250 * i));
        npy::ecrire_champ(&p, &champ_aleatoire(2, 40 + i), FormatChamp::C16, &libre()).unwrap();
        m.fichier(&p).unwrap();
    }
    fs::write(
        src.join("measurements.csv"),
        "measurement_index,total_steps,phase\n0,0,initial\n",
    )
    .unwrap();
    m.fichier(&src.join("measurements.csv")).unwrap();
    m.terminer(0).unwrap();

    // 2. copie sur le gamer, dans un répertoire de test dédié
    let distant = format!("kevinotron_dev/tests_fetch/ens_{}", std::process::id());
    let _nettoyage = NettoyageDistant(distant.clone());
    assert!(ssh(&format!("mkdir -p '{distant}'")).0);
    let o = Command::new("rsync")
        .args(["-a", "-e", "ssh -o BatchMode=yes"])
        .arg(format!("{}/", src.display()))
        .arg(format!("{CABLE}:{distant}/"))
        .output()
        .unwrap();
    assert!(
        o.status.success(),
        "rsync de préparation : {}",
        String::from_utf8_lossy(&o.stderr)
    );
    let (_, du) = ssh(&format!("du -sb '{distant}'"));
    let octets_distants: u64 = du.split_whitespace().next().unwrap().parse().unwrap();

    // 3. fetch nominal : sha256 vérifiés, contenu identique
    let d1 = sc.p("recu1");
    let r = recuperer(&format!("gamer:{distant}"), &d1, &libre(), CABLE).expect("fetch nominal");
    assert_eq!(r.sha_verifies, 4, "3 dumps + measurements.csv : {r:?}");
    assert_eq!(r.octets, octets_distants);
    assert!(r.secondes > 0.0);
    assert_eq!(
        empreinte_arbre(&d1),
        empreinte_arbre(&src),
        "le contenu rapatrié diffère de l'original"
    );
    assert!(d1.join("COMPLETE").exists());
    println!(
        "fetch nominal : {} octets en {:.2} s, {} sha256 vérifiés",
        r.octets, r.secondes, r.sha_verifies
    );

    // 4. un octet altéré À DISTANCE (après le manifeste) : erreur et fichier fautif supprimé
    let (ok, _) = ssh(&format!(
        "printf 'X' | dd of='{distant}/cfg_000250.npy' bs=1 seek=300 conv=notrunc 2>/dev/null"
    ));
    assert!(ok, "altération distante");
    let d2 = sc.p("recu2");
    let e = recuperer(&format!("gamer:{distant}"), &d2, &libre(), CABLE).unwrap_err();
    assert!(
        matches!(e, Erreur::Validation(_)),
        "attendu une erreur de validation : {e}"
    );
    assert_eq!(e.code_sortie(), 8);
    assert!(e.to_string().contains("cfg_000250.npy"), "{e}");
    assert!(
        !d2.join("cfg_000250.npy").exists(),
        "le fichier fautif doit être supprimé"
    );
    assert!(
        d2.join("cfg_000000.npy").exists() && d2.join("cfg_000500.npy").exists(),
        "les fichiers sains sont conservés"
    );
    // vérificateur du 04/10 : un ensemble partiel n'est JAMAIS marqué complet
    assert!(!d2.join("COMPLETE").exists(), "COMPLETE retiré après un fetch refusé");
    assert!(d2.join("FETCH_REFUSE.txt").exists(), "marqueur du refus");
    println!("octet altéré à distance : {e}");

    // 5. garde SSD sur la taille DISTANTE : un fichier creux de 3 Go (aucun bloc occupé sur le gamer)
    assert!(ssh(&format!("truncate -s 3G '{distant}/gros_creux.bin'")).0);
    let libre_go = GardeSsd::libre_go(&sc.0).unwrap();
    let d3 = sc.p("recu3");
    let g = GardeSsd::nouvelle(libre_go - 1.0); // 1 Go de marge seulement : 3 Go distants ne passent pas
    let e = recuperer(&format!("gamer:{distant}/gros_creux.bin"), &d3, &g, CABLE).unwrap_err();
    match &e {
        // 3 Gio (truncate -s 3G) = 3,0 Go : 1 Go = 1 Gio partout depuis le 04/10
        Erreur::SsdPlein { besoin_go, .. } => assert!((besoin_go - 3.0).abs() < 0.01, "besoin {besoin_go} Go"),
        autre => panic!("SsdPlein attendu, reçu {autre:?}"),
    }
    assert_eq!(e.code_sortie(), 7);
    assert!(!d3.exists(), "aucun octet copié, rien créé");
    // même chose sur l'ensemble entier (le fichier creux compte dans du -sb)
    let e = recuperer(&format!("gamer:{distant}"), &d3, &g, CABLE).unwrap_err();
    assert_eq!(e.code_sortie(), 7);
    assert!(!d3.exists());
    println!("garde SSD sur la taille distante : {e}");

    // 6. fichier seul : sha256 distant (sha256sum) contre local
    let d4 = sc.p("recu4");
    fs::create_dir_all(&d4).unwrap();
    let r = recuperer(&format!("gamer:{distant}/cfg_000000.npy"), &d4, &libre(), CABLE).unwrap();
    assert_eq!(r.sha_verifies, 1);
    assert_eq!(
        fs::read(d4.join("cfg_000000.npy")).unwrap(),
        fs::read(src.join("cfg_000000.npy")).unwrap()
    );

    // 7. sources invalides côté distant : erreur propre, rien de créé
    let d5 = sc.p("recu5");
    assert!(recuperer(&format!("gamer:{distant}/n_existe_pas"), &d5, &libre(), CABLE).is_err());
    assert!(!d5.exists());
    println!("PASS FETCH-1 par gamer-eth : sha256 vérifiés, octet altéré => erreur + fichier supprimé, garde sur la taille distante");
}
