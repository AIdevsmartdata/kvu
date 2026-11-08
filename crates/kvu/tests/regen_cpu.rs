// SPDX-License-Identifier: Apache-2.0
//! Rejeu `kvu regen` sur CPU (sans GPU, sans verrou) : une chaîne SU(2) L = 4 produite par le binaire lui-même, un manifeste
//! `kvu-regen/1` natif assemblé à la main, puis rejeu, mutant de graine (REGEN-3), originaux altérés, sorties interdites.
//! Les noms de tests commencent par `regen::` pour que `cargo test regen::` les sélectionne avec les tests de la bibliothèque.

#[path = "donnees/hermetique/support.rs"]
mod support;

mod regen {
    use kvu::heritage::manifest::sha256_fichier;
    use kvu::regen::rejeu::{rejouer, OptionsRejeu};
    use serde_json::{json, Value};
    use std::path::{Path, PathBuf};
    use std::process::Command;

    fn bin() -> PathBuf {
        PathBuf::from(env!("CARGO_BIN_EXE_kvu"))
    }

    fn tmp(nom: &str) -> PathBuf {
        // seuil de la garde SSD des tests (KVU_TEST_SSD_MIN_GO) pour `su2 gen` (sous-processus) et pour `rejouer` (ici)
        super::support::appliquer_seuil_ssd_des_tests();
        let base = std::env::var_os("KVU_SCRATCH").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
        let d = base.join(format!("regen_cpu_{nom}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn arbre(d: &Path) -> Vec<(String, u64)> {
        let mut v: Vec<(String, u64)> = kvu::regen::manifeste::parcourir(d)
            .unwrap()
            .into_iter()
            .map(|f| (f.relatif, f.octets))
            .collect();
        v.sort();
        v
    }

    /// Produit la chaîne d'origine (CPU) et un manifeste kvu-regen/1 natif ; rend (répertoire des originaux, manifeste).
    fn chaine_jouet(d: &Path, graine: u64) -> (PathBuf, PathBuf) {
        let orig = d.join("orig");
        let st = Command::new(bin())
            .args(["su2", "gen", "--backend", "cpu", "--precision", "f64", "--options", "ref-k1", "--L", "4", "--beta", "2.3"])
            .args(["--graine", &graine.to_string(), "--start", "hot", "--therm", "20", "--nmeas", "5", "--skip", "1", "--nor", "4"])
            .args(["--dump-every", "1", "--outdir"])
            .arg(&orig)
            .output()
            .unwrap();
        assert!(st.status.success(), "{}", String::from_utf8_lossy(&st.stderr));
        let mut fichiers = Vec::new();
        for nom in ["cfg_000001.npy", "cfg_000002.npy", "cfg_000003.npy", "cfg_000004.npy", "cfg_000005.npy", "measurements.csv", "therm.csv"] {
            let p = orig.join(nom);
            fichiers.push(json!({"nom": nom, "sha256": sha256_fichier(&p).unwrap(), "octets": std::fs::metadata(&p).unwrap().len()}));
        }
        let sha_bin = sha256_fichier(&bin()).unwrap();
        let m = json!({
            "schema": "kvu-regen/1",
            "origine": {"schema": "kvu-regen/1", "manifeste": "test", "sha256_manifeste": "a".repeat(64)},
            "binaire_sha256": sha_bin, "binaire_provenance": "manifeste", "git_head": "test", "source_hash": "b".repeat(64),
            "noyaux_embarques": [], "modules_compiles": [], "backend": "cpu", "arch": null, "versions": {},
            "machine": {"nom": "inconnue", "cpu": null, "gpu": null, "ram": null},
            "graines": {"cle": graine, "etiquettes": ["test"]},
            "parametres": {"groupe": "SU(2)", "L": 4, "beta": 2.3, "start": "hot", "therm": 20, "nmeas": 5, "skip": 1, "nor": 4,
                           "precision": "f64", "options": "ref-k1", "dump_every": 1, "bord": false},
            "commande": ["su2", "gen", "--backend", "cpu", "--precision", "f64", "--options", "ref-k1", "--L", "4", "--beta", "2.3",
                         "--graine", graine.to_string(), "--start", "hot", "--therm", "20", "--nmeas", "5", "--skip", "1", "--nor", "4",
                         "--dump-every", "1"],
            "entrees": [], "fichiers": fichiers, "fichiers_hachage": "manifeste", "duree_s": 0.5, "statut": "COMPLETE",
            "classe_regen": "C", "regen_verifiee": {"date": null, "machine": null, "resultat": "non_verifiee"}
        });
        let mp = d.join("M.json");
        std::fs::write(&mp, serde_json::to_string_pretty(&m).unwrap()).unwrap();
        (orig, mp)
    }

    fn opts(sortie: PathBuf) -> OptionsRejeu {
        OptionsRejeu { sortie, executant: Some(bin()), ..Default::default() }
    }

    #[test]
    fn regen_cpu_rejeu_identique_et_copie_du_manifeste() {
        let d = tmp("ok");
        let (orig, mp) = chaine_jouet(&d, 7);
        let avant = arbre(&orig);
        let r = rejouer(&mp, &opts(d.join("sortie"))).unwrap();
        assert!(r.tout_identique(), "{:?}", r.fichiers);
        assert_eq!((r.identiques, r.fichiers.len()), (7, 7));
        let copie: Value = serde_json::from_slice(&std::fs::read(&r.manifeste_regen).unwrap()).unwrap();
        assert_eq!(copie["regen_verifiee"]["resultat"], "identique");
        assert!(copie["regen_verifiee"]["date"].as_str().unwrap().ends_with('Z'));
        // le manifeste d'origine n'est pas modifié ; la copie est valide au schéma
        let origine: Value = serde_json::from_slice(&std::fs::read(&mp).unwrap()).unwrap();
        assert_eq!(origine["regen_verifiee"]["resultat"], "non_verifiee");
        assert!(kvu::regen::schema::valider_manifeste(&copie).is_empty());
        assert_eq!(arbre(&orig), avant, "les originaux ne bougent pas");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn regen_cpu_mutant_graine_detecte() {
        let d = tmp("mut");
        let (_orig, mp) = chaine_jouet(&d, 7);
        let mut o = opts(d.join("sortie"));
        o.mutant_graine = true;
        let r = rejouer(&mp, &o).unwrap();
        assert!(!r.tout_identique());
        assert!(r.identiques < r.fichiers.len());
        let copie: Value = serde_json::from_slice(&std::fs::read(&r.manifeste_regen).unwrap()).unwrap();
        assert_eq!(copie["regen_verifiee"]["resultat"], "different");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn regen_cpu_sha_du_manifeste_altere_detecte() {
        let d = tmp("alt");
        let (_orig, mp) = chaine_jouet(&d, 7);
        let mut m: Value = serde_json::from_slice(&std::fs::read(&mp).unwrap()).unwrap();
        m["fichiers"][2]["sha256"] = json!("0".repeat(64));
        std::fs::write(&mp, serde_json::to_string(&m).unwrap()).unwrap();
        let r = rejouer(&mp, &opts(d.join("sortie"))).unwrap();
        assert!(!r.tout_identique());
        assert_eq!(r.identiques, r.fichiers.len() - 1);
        assert_eq!(r.fichiers[2].etat, "DIFFERENT");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn regen_cpu_binaire_non_admis_est_refuse_sans_rien_ecrire() {
        let d = tmp("bin");
        let (_orig, mp) = chaine_jouet(&d, 7);
        let mut m: Value = serde_json::from_slice(&std::fs::read(&mp).unwrap()).unwrap();
        m["binaire_sha256"] = json!("c".repeat(64));
        std::fs::write(&mp, serde_json::to_string(&m).unwrap()).unwrap();
        let e = rejouer(&mp, &opts(d.join("sortie"))).unwrap_err();
        assert!(e.to_string().contains("binaire non admis"), "{e}");
        assert_eq!(e.code_sortie(), 2);
        assert!(!d.join("sortie").exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn regen_cpu_sortie_existante_ou_sous_l_ensemble_refusee() {
        let d = tmp("sortie");
        let (orig, mp) = chaine_jouet(&d, 7);
        std::fs::create_dir_all(d.join("existe")).unwrap();
        assert!(rejouer(&mp, &opts(d.join("existe"))).unwrap_err().to_string().contains("REFUS D'ÉCRASER"));
        // sortie sous le répertoire des originaux (comparer explicite)
        let mut o = opts(orig.join("sortie_interdite"));
        o.comparer = Some(orig.clone());
        let e = rejouer(&mp, &o).unwrap_err().to_string();
        assert!(e.contains("REFUS"), "{e}");
        assert!(!orig.join("sortie_interdite").exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn regen_cpu_manifeste_invalide_au_schema_refuse() {
        let d = tmp("inv");
        let (_orig, mp) = chaine_jouet(&d, 7);
        let mut m: Value = serde_json::from_slice(&std::fs::read(&mp).unwrap()).unwrap();
        m.as_object_mut().unwrap().remove("graines");
        std::fs::write(&mp, serde_json::to_string(&m).unwrap()).unwrap();
        let e = rejouer(&mp, &opts(d.join("sortie"))).unwrap_err();
        assert!(e.to_string().contains("invalide au schéma"), "{e}");
        assert!(!d.join("sortie").exists());
        let _ = std::fs::remove_dir_all(&d);
    }
}
