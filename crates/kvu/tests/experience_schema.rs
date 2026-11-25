// SPDX-License-Identifier: Apache-2.0
//! E-SCHEMA : rejet des structures et des statuts interdits par le binaire.

use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static COMPTEUR: AtomicU64 = AtomicU64::new(0);

struct Dossier(PathBuf);

impl Dossier {
    fn nouveau() -> Self {
        let chemin = std::env::temp_dir().join(format!(
            "kvu-experience-schema-{}-{}",
            std::process::id(),
            COMPTEUR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&chemin).unwrap();
        let dossier = Self(chemin);
        fs::write(
            dossier.0.join("PREREG.md"),
            "# Critère gelé\nSeuil déclaré.\nFin.\n",
        )
        .unwrap();
        let sortie = dossier.kvu(&[
            "init",
            "--id",
            "synthetique",
            "--titre",
            "Expérience synthétique",
            "--question",
            "Question du témoin",
            "--prereg",
            "PREREG.md",
            "--gele-le",
            "2001-09-09T01:46:40Z",
            "--machine-principale",
            "principale",
        ]);
        assert_eq!(
            sortie.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&sortie.stderr)
        );
        dossier
    }

    fn kvu(&self, args: &[&str]) -> Output {
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

    fn lire(&self) -> Value {
        serde_json::from_slice(&fs::read(self.0.join("EXPERIENCE.json")).unwrap()).unwrap()
    }

    fn ecrire(&self, v: &Value) {
        fs::write(
            self.0.join("EXPERIENCE.json"),
            serde_json::to_vec_pretty(v).unwrap(),
        )
        .unwrap();
    }

    fn rouge_v1(&self) {
        let sortie = self.kvu(&["verifier"]);
        assert_eq!(sortie.status.code(), Some(8), "{sortie:?}");
        assert!(
            String::from_utf8_lossy(&sortie.stdout).contains("ROUGE V1"),
            "{sortie:?}"
        );
    }
}

impl Drop for Dossier {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn e_schema_sans_prereg() {
    let d = Dossier::nouveau();
    let mut v = d.lire();
    v.as_object_mut().unwrap().remove("prereg");
    d.ecrire(&v);
    d.rouge_v1();
}

#[test]
fn e_schema_verdict_hors_liste() {
    let d = Dossier::nouveau();
    let mut v = d.lire();
    v["verdict"]["statut"] = json!("etabli");
    d.ecrire(&v);
    d.rouge_v1();
}

#[test]
fn e_schema_statut_lean_interdit() {
    let d = Dossier::nouveau();
    let mut v = d.lire();
    v["carnet"] = json!([{
        "type": "fait", "id": "FAI-temoin", "titre": "Lecture",
        "statut": "lean", "enonce": "Observation synthétique",
        "sources": ["PREREG.md:1"], "liens": [], "date": "2001-09-09T02:00:00Z"
    }]);
    d.ecrire(&v);
    d.rouge_v1();
}

#[test]
fn schema_embarque_identique_et_temoins_valides() {
    let d = Dossier::nouveau();
    let embarque = kvu::experience::schema::SCHEMA_TEXTE;
    let copie = include_str!("../../../schemas/kvu-experience-1.schema.json");
    assert_eq!(embarque.as_bytes(), copie.as_bytes());
    let schema: Value = serde_json::from_str(embarque).unwrap();
    assert!(kvu::regen::schema::valider(&schema, &d.lire())
        .unwrap()
        .is_empty());
    assert!(kvu::experience::schema::valider_experience(&d.lire()).is_empty());
    let sortie = d.kvu(&["verifier", "--json"]);
    assert_eq!(sortie.status.code(), Some(0), "{sortie:?}");
    let constats: Value = serde_json::from_slice(&sortie.stdout).unwrap();
    // Onze contrôles ; l'horloge imposée des tests ajoute un AVERTISSEMENT (V4) en plus.
    let mut controles: Vec<u64> = constats
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["controle"].as_u64().unwrap())
        .collect();
    controles.dedup();
    assert_eq!(controles, (1..=11).collect::<Vec<u64>>());
    assert!(constats
        .as_array()
        .unwrap()
        .iter()
        .all(|c| c["ok"] == true || c["avertissement"] == true));
}
