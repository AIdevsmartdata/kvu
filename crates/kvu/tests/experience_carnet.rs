// SPDX-License-Identifier: Apache-2.0
//! E-IDEMP et contrat du carnet sur un dossier synthétique jetable.

use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static COMPTEUR: AtomicU64 = AtomicU64::new(0);

struct CarnetTest(PathBuf);

impl CarnetTest {
    fn nouveau() -> Self {
        let p = std::env::temp_dir().join(format!(
            "kvu-experience-carnet-{}-{}",
            std::process::id(),
            COMPTEUR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        fs::write(
            p.join("PREREG.md"),
            "# Hypothèse\nCritère gelé.\nTroisième ligne.\n",
        )
        .unwrap();
        let e = Self(p);
        e.ok(&[
            "init",
            "--id",
            "carnet",
            "--titre",
            "Titre : \"cité\"",
            "--question",
            "Une question synthétique",
            "--prereg",
            "PREREG.md",
            "--gele-le",
            "2001-09-09T01:46:40Z",
            "--chemin-projet",
            "chantier/temoin",
            "--machine-principale",
            "principale",
        ]);
        e
    }

    fn lancer(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_kvu"))
            .arg("experience")
            .args(args)
            .arg("--racine")
            .arg(&self.0)
            .env("KVU_EXPERIENCE_HORLOGE", "2001-09-09T02:00:00Z")
            .current_dir(&self.0)
            .output()
            .unwrap()
    }

    fn ok(&self, args: &[&str]) -> Output {
        let sortie = self.lancer(args);
        assert_eq!(sortie.status.code(), Some(0), "{args:?} : {sortie:?}");
        sortie
    }

    fn proposer(&self) {
        self.ok(&[
            "proposer",
            "resultat",
            "RES-temoin",
            "--titre",
            "Résultat : \"témoin\"",
            "--statut",
            "numerique",
            "--enonce",
            "Lecture synthétique.",
            "--source",
            "PREREG.md:1-3",
            "--lien",
            "EXP-carnet",
            "--tomberait",
            "Si le contrôle échoue : relire.",
        ]);
    }

    fn lire(&self) -> Value {
        serde_json::from_slice(&fs::read(self.0.join("EXPERIENCE.json")).unwrap()).unwrap()
    }

    fn ecrire(&self, v: &Value) {
        let mut octets = serde_json::to_vec_pretty(v).unwrap();
        octets.push(b'\n');
        fs::write(self.0.join("EXPERIENCE.json"), octets).unwrap();
    }

    fn instantane(&self) -> BTreeMap<String, Vec<u8>> {
        let mut resultat = BTreeMap::new();
        photographier(
            &self.0.join("carnet"),
            &self.0.join("carnet"),
            &mut resultat,
        );
        resultat
    }
}

impl Drop for CarnetTest {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn photographier(racine: &Path, dir: &Path, resultat: &mut BTreeMap<String, Vec<u8>>) {
    for entree in fs::read_dir(dir).unwrap() {
        let entree = entree.unwrap();
        let p = entree.path();
        let relatif = p.strip_prefix(racine).unwrap().to_str().unwrap().to_owned();
        if entree.file_type().unwrap().is_dir() {
            resultat.insert(format!("{relatif}/"), Vec::new());
            photographier(racine, &p, resultat);
        } else {
            resultat.insert(relatif, fs::read(p).unwrap());
        }
    }
}

#[test]
fn e_idemp_deux_exports_puis_version_de_proposition() {
    let e = CarnetTest::nouveau();
    e.proposer();
    e.ok(&["carnet", "--vers", "carnet"]);
    let premier = e.instantane();
    e.ok(&["carnet", "--vers", "carnet"]);
    assert_eq!(e.instantane(), premier);
    let ancien = fs::read(e.0.join("carnet/resultats/RES-temoin.md")).unwrap();
    let mut v = e.lire();
    v["carnet"][0]["enonce"] = json!("Le contenu de la proposition a changé.");
    e.ecrire(&v);
    e.ok(&["carnet", "--vers", "carnet"]);
    assert_eq!(
        fs::read(e.0.join("carnet/resultats/RES-temoin.md")).unwrap(),
        ancien
    );
    let nouveau = fs::read_to_string(e.0.join("carnet/resultats/RES-temoin-v2.md")).unwrap();
    assert!(nouveau.contains("id: RES-temoin-v2\n"));
    assert!(nouveau.contains("remplace: RES-temoin\n"));
    assert!(nouveau.contains("Le contenu de la proposition a changé."));
    let versionne = e.instantane();
    e.ok(&["carnet", "--vers", "carnet"]);
    assert_eq!(e.instantane(), versionne);
    assert!(!e.0.join("carnet/resultats/RES-temoin-v3.md").exists());
}

#[test]
fn entete_sources_plages_et_corps_conformes_au_carnet() {
    let e = CarnetTest::nouveau();
    e.proposer();
    e.ok(&["carnet", "--vers", "carnet"]);
    for dossier in [
        "experiences",
        "resultats",
        "erreurs",
        "decisions",
        "hypotheses",
        "pistes",
        "faits",
    ] {
        assert!(e.0.join("carnet").join(dossier).is_dir());
    }
    let contenu = fs::read_to_string(e.0.join("carnet/resultats/RES-temoin.md")).unwrap();
    assert!(contenu.starts_with("---\n"));
    let (entete, corps) = contenu[4..].split_once("\n---\n").unwrap();
    for cle in [
        "id",
        "titre",
        "date",
        "type",
        "statut",
        "sources",
        "liens",
        "experience",
        "manifeste_sha256",
        "tomberait",
    ] {
        assert!(
            entete.lines().any(|l| l.starts_with(&format!("{cle}:"))),
            "clé {cle} : {entete}"
        );
    }
    assert!(entete.contains("titre: \"Résultat : \\\"témoin\\\"\""));
    assert!(entete.contains("date: \"2001-09-09T02:00\""));
    assert!(entete.contains("chantier/temoin/PREREG.md:1"));
    assert!(entete.contains("chantier/temoin/PREREG.md:3"));
    assert!(!entete.contains("PREREG.md:1-3"));
    assert!(corps.starts_with("# RES-temoin :"));
    assert!(corps.trim().lines().count() >= 3);
    let experience = fs::read_to_string(e.0.join("carnet/experiences/EXP-carnet.md")).unwrap();
    assert!(experience.contains("statut: \"prereg\""));
    assert!(experience.contains("date: \"2001-09-09T01:46\""));
}

#[test]
fn changement_de_seule_empreinte_ne_cree_aucune_version() {
    let e = CarnetTest::nouveau();
    e.proposer();
    e.ok(&["carnet", "--vers", "carnet"]);
    let premier = e.instantane();
    let mut v = e.lire();
    v["index"] = json!({"note": "Métadonnée sans changement du contenu exporté"});
    e.ecrire(&v);
    e.ok(&["carnet", "--vers", "carnet"]);
    assert_eq!(e.instantane(), premier);
}

#[test]
fn sources_hors_bornes_et_liens_inconnus_refuses() {
    let e = CarnetTest::nouveau();
    e.proposer();
    let mut v = e.lire();
    v["carnet"][0]["sources"] = json!(["PREREG.md:1-99"]);
    e.ecrire(&v);
    assert_eq!(e.lancer(&["verifier"]).status.code(), Some(8));
    assert_eq!(
        e.lancer(&["carnet", "--vers", "carnet"]).status.code(),
        Some(8)
    );
    assert!(!e.0.join("carnet").exists());
    v["carnet"][0]["sources"] = json!(["PREREG.md:1"]);
    v["carnet"][0]["liens"] = json!(["RES-inconnu"]);
    e.ecrire(&v);
    assert_eq!(
        e.lancer(&["carnet", "--vers", "carnet"]).status.code(),
        Some(8)
    );
    assert!(!e.0.join("carnet").exists());
}
