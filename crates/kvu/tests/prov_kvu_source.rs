// SPDX-License-Identifier: Apache-2.0
//! Confrontation de la recette Rust au recalcul Python indépendant.

use kvu::stockage::recette::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct DepotJouet(PathBuf);

impl DepotJouet {
    fn nouveau(nom: &str) -> Self {
        let racine = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("{nom}_{}", std::process::id()));
        fs::create_dir_all(racine.parent().expect("parent du dépôt jouet"))
            .expect("création du dossier des tests");
        fs::create_dir(&racine).expect("création du dépôt jouet propre au test");
        let depot = Self(racine);
        for (chemin, contenu) in [
            ("Cargo.lock", "# verrou jouet\nversion = 4\n"),
            ("Cargo.toml", "[workspace]\nmembers = [\"crates/kvu\"]\n"),
            ("crates/kvu/Cargo.toml", "[package]\nname = \"kvu\"\nversion = \"0.1.0\"\n"),
            ("crates/kvu/build.rs", "fn main() {}\n"),
            ("rust-toolchain.toml", "[toolchain]\nchannel = \"1.97.0\"\n"),
            ("crates/kvu/src/lib.rs", "pub fn jouet() {}\n"),
            ("crates/kvu/src/heritage/ancien.rs", "// source historique couverte\n"),
            ("crates/kvu/noyaux/su2_commun.h", "#define JOUET 0\n"),
            ("crates/kvu/noyaux/sous/reduction.h", "#define REDUCTION 1\n"),
            ("crates/kvu/noyaux/é.h", "// nom non ASCII\n"),
            ("crates/kvu/tests/x.rs", "// hors recette, avant\n"),
            ("README.md", "Dépôt jouet, avant.\n"),
        ] {
            depot.ecrire(chemin, contenu.as_bytes());
        }
        depot.ecrire(
            "crates/kvu/src/stockage/recette.rs",
            &fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/stockage/recette.rs"))
                .expect("lecture de la recette réelle à copier"),
        );
        depot
    }

    fn racine(&self) -> &Path {
        &self.0
    }

    fn ecrire(&self, relatif: &str, contenu: &[u8]) {
        let chemin = self.0.join(relatif);
        fs::create_dir_all(chemin.parent().expect("parent du fichier jouet"))
            .expect("création du sous-dossier jouet");
        fs::write(chemin, contenu).expect("écriture du fichier jouet");
    }

    fn modifier_un_octet(&self, relatif: &str) -> Vec<u8> {
        let avant = fs::read(self.0.join(relatif)).expect("lecture avant mutation");
        let mut apres = avant.clone();
        *apres.last_mut().expect("fichier jouet non vide") ^= 1;
        self.ecrire(relatif, &apres);
        avant
    }

    fn supprimer(self) {
        fs::remove_dir_all(&self.0).expect("suppression du dépôt jouet après le test");
    }
}

impl Drop for DepotJouet {
    fn drop(&mut self) {
        // Nettoyage aussi lors d'une panique ; le chemin appartient à ce test.
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn python(racine: &Path, options: &[&str]) -> Output {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("validation/historique/source_hash_kvu.py");
    Command::new("python3")
        .arg("-I")
        .arg(script)
        .arg(racine)
        .args(options)
        .output()
        .expect("exécution du recalcul Python indépendant")
}

fn comparer(racine: &Path) -> (String, usize) {
    let rust = hash_kvu_source_1(racine).expect("calcul Rust de kvu-source/1");
    let sortie = python(racine, &[]);
    assert!(
        sortie.status.success(),
        "échec Python : {}",
        String::from_utf8_lossy(&sortie.stderr)
    );
    let empreinte = String::from_utf8(sortie.stdout).expect("empreinte Python UTF-8");
    assert_eq!(empreinte.trim().len(), 64, "taille de l'empreinte Python");
    assert!(empreinte.trim().bytes().all(|b| b.is_ascii_hexdigit()), "empreinte hexadécimale");
    assert_eq!(rust.0, empreinte.trim(), "empreintes Rust et Python");
    assert_eq!(
        String::from_utf8(sortie.stderr).expect("effectif Python UTF-8").trim(),
        format!("{} fichiers", rust.1),
        "effectifs Rust et Python"
    );
    rust
}

#[test]
fn kvu_source_1_arbre_jouet_rust_egal_python() {
    let depot = DepotJouet::nouveau("kvu_source_1_arbre_jouet_rust_egal_python");
    let (_, nombre) = comparer(depot.racine());
    let rust = fichiers_kvu_source_1(depot.racine()).expect("liste Rust");
    let sortie = python(depot.racine(), &["--liste"]);
    assert!(sortie.status.success(), "liste Python : {}", String::from_utf8_lossy(&sortie.stderr));
    let texte = String::from_utf8(sortie.stdout).expect("liste Python UTF-8");
    let liste_python: Vec<String> = texte.lines().map(str::to_owned).collect();
    assert_eq!(rust, liste_python, "listes triées Rust et Python");
    assert_eq!(nombre, 11, "effectif connu du dépôt jouet");
    assert_eq!(rust.len(), nombre, "effectif et longueur de la liste");
    assert!(rust.windows(2).all(|p| p[0] < p[1]), "tri strict sans doublon");
    for attendu in [
        "crates/kvu/noyaux/é.h",
        "crates/kvu/src/heritage/ancien.rs",
        "crates/kvu/src/stockage/recette.rs",
    ] {
        assert!(rust.iter().any(|p| p == attendu), "chemin couvert : {attendu}");
    }
    depot.supprimer();
    println!("PASS kvu_source_1_arbre_jouet_rust_egal_python");
}

#[test]
fn kvu_source_1_sensibilite_et_exclusions() {
    let depot = DepotJouet::nouveau("kvu_source_1_sensibilite_et_exclusions");
    let reference = comparer(depot.racine());
    for chemin in [
        "crates/kvu/noyaux/su2_commun.h",
        "crates/kvu/src/lib.rs",
        "rust-toolchain.toml",
        "Cargo.lock",
    ] {
        let avant = depot.modifier_un_octet(chemin);
        let mutant = comparer(depot.racine());
        assert_ne!(reference.0, mutant.0, "mutation détectée : {chemin}");
        assert_eq!(reference.1, mutant.1, "effectif après mutation : {chemin}");
        depot.ecrire(chemin, &avant);
    }
    depot.ecrire("crates/kvu/noyaux/ajout.h", b"// noyau ajoute\n");
    let ajout = comparer(depot.racine());
    assert_ne!(reference.0, ajout.0, "ajout d'un noyau détecté");
    assert_eq!(ajout.1, reference.1 + 1, "effectif avec le noyau ajouté");
    fs::remove_file(depot.racine().join("crates/kvu/noyaux/ajout.h"))
        .expect("retrait du noyau ajouté");
    assert_eq!(reference, comparer(depot.racine()), "retour à la référence");
    for chemin in ["crates/kvu/tests/x.rs", "README.md"] {
        depot.modifier_un_octet(chemin);
        assert_eq!(reference, comparer(depot.racine()), "fichier hors recette : {chemin}");
    }
    depot.supprimer();
    println!("PASS kvu_source_1_sensibilite_et_exclusions");
}

#[test]
fn kvu_source_1_mutant_noyau_detecte() {
    let depot = DepotJouet::nouveau("kvu_source_1_mutant_noyau_detecte");
    let reference = comparer(depot.racine());
    let noyau = "crates/kvu/noyaux/su2_commun.h";
    depot.modifier_un_octet(noyau);
    let mutant = comparer(depot.racine());
    assert_ne!(reference.0, mutant.0, "le noyau mutant change l'empreinte");
    assert!(
        fichiers_kvu_source_1(depot.racine()).expect("liste après mutation")
            .iter().any(|p| p == noyau),
        "le noyau mutant reste dans la recette"
    );
    fs::remove_file(depot.racine().join("Cargo.lock")).expect("retrait du verrou obligatoire");
    assert!(fichiers_kvu_source_1(depot.racine()).is_err(), "liste Rust : verrou absent refusé");
    assert!(hash_kvu_source_1(depot.racine()).is_err(), "empreinte Rust : verrou absent refusé");
    let sortie = python(depot.racine(), &[]);
    assert!(!sortie.status.success(), "Python doit refuser le verrou absent");
    assert_eq!(sortie.status.code(), Some(2), "code Python pour un fichier manquant");
    assert!(sortie.stdout.is_empty(), "aucune empreinte en cas d'erreur");
    assert!(String::from_utf8_lossy(&sortie.stderr).contains("Cargo.lock"), "diagnostic du verrou absent");
    depot.supprimer();
    println!("PASS kvu_source_1_mutant_noyau_detecte");
}
