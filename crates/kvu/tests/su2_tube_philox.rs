//! Module tube (EXPÉRIMENTAL), relecture c16b, objection 14 : les couples (clé Philox, compteur) RÉELLEMENT passés à Philox
//! par le module pendant un run (initialisation, tirages et rejets du bain thermique compris) sont deux à deux distincts :
//! deux flux censés être indépendants (tâches β × réplique × secteur d'`apparie`, chaînes d'un lot `gen --cotes`) ne
//! partagent jamais un nombre aléatoire. Les runs sont exécutés EN PROCESSUS (même code que la ligne de commande), sous le
//! traceur `reseau::trace_philox` ; ce fichier ne contient qu'un test, car le traceur est global au processus.

use kvu::su2::tube::reseau::trace_philox::{self, Appel};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

fn dossier(nom: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("kvu_su2_tube_philox_{nom}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Exécute `kvu <argv>` en processus sous le traceur et rend les appels Philox notés.
fn lancer(argv: &[&str]) -> Vec<Appel> {
    let m = kvu::su2::cli::commandes()
        .into_iter()
        .next()
        .unwrap()
        .try_get_matches_from(argv)
        .unwrap_or_else(|e| panic!("{argv:?} : {e}"));
    trace_philox::demarrer();
    let r = kvu::su2::cli::executer("su2", &m);
    let notes = trace_philox::arreter();
    match r {
        Ok(0) => notes,
        Ok(rc) => panic!("{argv:?} : code {rc}"),
        Err(e) => panic!("{argv:?} : {e}"),
    }
}

/// Aucun couple (clé, compteur) n'est employé deux fois ; rend le nombre de clés distinctes.
fn disjoints(quoi: &str, notes: &[Appel]) -> usize {
    let mut vus = HashSet::with_capacity(notes.len());
    for n in notes {
        assert!(
            vus.insert(*n),
            "{quoi} : couple (clé, compteur) employé deux fois : {n:?}"
        );
    }
    notes
        .iter()
        .map(|n| (n.0, n.1))
        .collect::<HashSet<_>>()
        .len()
}

#[test]
fn couples_philox_reellement_employes_disjoints_entre_flux() {
    // un seul test dans ce binaire : les variables d'environnement ne sont lues par aucun autre fil
    std::env::set_var("KEVINOTRON_RAM_MIN_GO", "0");
    std::env::set_var("KEVINOTRON_SSD_MIN_GO", "0");
    std::env::set_var("KEVINOTRON_VERROU", "ignorer");
    let d = dossier("runs");
    // graines : 0, 1 (défaut), 2⁶⁴ − 1, 7919 (le pas de la dérivation des graines de tâche)
    for (i, g) in ["0", "1", "18446744073709551615", "7919"]
        .iter()
        .enumerate()
    {
        for depart in ["chaud", "minimum"] {
            // apparie : 2 β (régimes de Creutz, α < 2, et de Kennedy-Pendleton) × 2 répliques × 2 secteurs = 8 tâches
            let out = d.join(format!("a{i}_{depart}"));
            let o = out.to_string_lossy().into_owned();
            let notes = lancer(&[
                "su2",
                "tube",
                "apparie",
                "--cotes",
                "2,2,2,2",
                "--beta",
                "0.5,4",
                "--repliques",
                "2",
                "--therm",
                "3",
                "--nmeas",
                "20",
                "--series",
                "non",
                "--graine",
                g,
                "--depart",
                depart,
                "--outdir",
                &o,
            ]);
            // 8 tâches × 23 balayages × 64 liens au moins (un tirage par lien, plus les rejets)
            assert!(
                notes.len() >= 8 * 23 * 64,
                "apparie graine {g} {depart} : {} appels seulement",
                notes.len()
            );
            let cles = disjoints(&format!("apparie graine {g} {depart}"), &notes);
            assert!(
                cles >= 8,
                "apparie graine {g} {depart} : {cles} clés pour 8 tâches"
            );
        }
        // gen --cotes : lot de 3 chaînes (graine + i), départ chaud (initialisation tracée)
        let out = d.join(format!("g{i}"));
        let o = out.to_string_lossy().into_owned();
        let notes = lancer(&[
            "su2",
            "gen",
            "--cotes",
            "2,2,2,2",
            "--backend",
            "cpu",
            "--start",
            "hot",
            "--beta",
            "0.5,4,1",
            "--graine",
            g,
            "--graine-pas",
            "1",
            "--chaines",
            "3",
            "--therm",
            "3",
            "--nmeas",
            "5",
            "--nor",
            "1",
            "--outdir",
            &o,
        ]);
        assert!(
            notes.len() >= 3 * 8 * 64,
            "gen graine {g} : {} appels seulement",
            notes.len()
        );
        assert_eq!(
            disjoints(&format!("gen graine {g}"), &notes),
            3,
            "gen graine {g} : une clé par chaîne"
        );
    }
}
