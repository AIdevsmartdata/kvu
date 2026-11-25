//! Couples (clé, compteur) RÉELLEMENT consommés par le générateur du module, commande par commande (relecture c19b,
//! majeure 12) : journal `intrication::journal_flux` (départ chaud et chaque tirage du bain thermique, rejets compris), sur
//! plusieurs balayages par chaîne. Le journal est global : ce fichier est seul dans son processus et ne contient qu'UN test,
//! qui exécute les commandes l'une après l'autre par `cli::executer` (le chemin de `kvu intrication`).
//! Exclus et nommés : les RÉEMPLOIS VOLONTAIRES de la commande `portes`, où une porte et son mutant (ou deux portes de
//! comparaison) rejouent la même configuration avec les mêmes flux historiques (liste `REEMPLOIS_PORTES`).

use kvu::intrication::{cli, journal_flux, USAGE_HB, USAGE_INIT};

/// Clés (graines historiques) réemployées volontairement par la commande `portes` : (graine, portes concernées).
const REEMPLOIS_PORTES: &[(u64, &str)] = &[
    (77, "(b) et son mutant f_b"),
    (77 ^ 0x5DEE_CE66_D1CE_4E5B, "(b) et son mutant f_b"),
    (5, "(c) mutant et (c) sans identification ; (d1) largeur 2"),
    (
        5u64.rotate_left(17),
        "(c) mutant et (c) sans identification",
    ),
    (4, "(d1) largeur 1 et son mutant"),
    (6, "(d1) largeur 2 à α = 0 et α = 1 (et largeur L/2)"),
    (21, "(d3) et son mutant"),
    (21 ^ 0xABCD, "(d3) et son mutant"),
];

/// Exécute `intrication args…` dans ce processus, journal actif ; rend le code (0 exigé, ou 8 si `accepter_8` : désaccord
/// statistique |z| ≥ 4 des deux routes de `doublement`, sans effet sur les flux) et les couples consommés.
fn executer_rc(args: &[&str], accepter_8: bool) -> Vec<[u32; 6]> {
    let m = cli::commande().try_get_matches_from(args).unwrap();
    journal_flux::activer();
    let rc = cli::executer("intrication", &m);
    let j = journal_flux::prendre();
    let rc = rc.unwrap();
    assert!(rc == 0 || (accepter_8 && rc == 8), "{args:?} : code {rc}");
    j
}

fn executer(args: &[&str]) -> Vec<[u32; 6]> {
    executer_rc(args, false)
}

/// Couples présents au moins deux fois.
fn doublons(j: &[[u32; 6]]) -> Vec<[u32; 6]> {
    let mut v = j.to_vec();
    v.sort_unstable();
    let mut d: Vec<[u32; 6]> = v
        .windows(2)
        .filter(|w| w[0] == w[1])
        .map(|w| w[0])
        .collect();
    d.dedup();
    d
}

/// Le journal contient des tirages de départ chaud ET de bain thermique (usage, bits 30-31 du mot 3 en mode à domaines).
fn usages(j: &[[u32; 6]]) -> (usize, usize) {
    (
        j.iter().filter(|c| c[5] >> 30 == USAGE_INIT).count(),
        j.iter().filter(|c| c[5] >> 30 == USAGE_HB).count(),
    )
}

#[test]
fn couples_consommes_distincts_par_commande() {
    let base = std::env::temp_dir().join(format!("kvu_intrication_flux_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let sortie = |n: &str| base.join(n).to_str().unwrap().to_string();
    // therm 2 + mes 4 × sep 1 = 6 balayages de bain thermique par chaîne
    let chaine = [
        "--therm", "2", "--mes", "4", "--sep", "1", "--blocs", "2", "--or", "1",
    ];
    let mut union: Vec<[u32; 6]> = Vec::new();
    for graine in ["0", "1", "1071", "18446744073709551615"] {
        // balayage : 3 points α × 2 répliques, L = T = 2 (128 liens)
        let s = sortie(&format!("b{graine}"));
        let mut a = vec![
            "intrication",
            "balayage",
            "--l",
            "2",
            "--t",
            "2",
            "--l0",
            "0",
            "--l1",
            "1",
            "--beta",
            "1.0",
            "--nalpha",
            "3",
        ];
        a.extend(["--graine", graine, "--sortie", s.as_str()]);
        a.extend(chaine);
        let j = executer(&a);
        let (ni, nh) = usages(&j);
        assert!(
            ni >= 3 * 2 * 128 && nh >= 3 * 6 * 128,
            "balayage, graine {graine} : {ni} départs, {nh} tirages"
        );
        assert!(
            doublons(&j).is_empty(),
            "balayage, graine {graine} : couple réemployé {:?}",
            doublons(&j)[0]
        );
        union.extend(j);
        // courbe : trois largeurs (champs du flux), L = 4, T = 2
        let s = sortie(&format!("c{graine}"));
        let mut a = vec![
            "intrication",
            "courbe",
            "--l",
            "4",
            "--t",
            "2",
            "--largeurs",
            "0,1,2",
            "--beta",
            "1.0",
            "--nalpha",
            "3",
        ];
        a.extend(["--graine", graine, "--sortie", s.as_str()]);
        a.extend(chaine);
        let j = executer(&a);
        assert!(usages(&j).0 > 0 && usages(&j).1 > 0);
        assert!(
            doublons(&j).is_empty(),
            "courbe, graine {graine} : couple réemployé {:?}",
            doublons(&j)[0]
        );
        // doublement : routes α et β (géométries vide et pleine)
        let s = sortie(&format!("d{graine}"));
        let mut a = vec![
            "intrication",
            "doublement",
            "--l",
            "2",
            "--t",
            "2",
            "--beta",
            "1.0",
            "--nalpha",
            "3",
            "--nbeta",
            "3",
        ];
        a.extend(["--graine", graine, "--sortie", s.as_str()]);
        a.extend(chaine);
        let j = executer_rc(&a, true);
        assert!(usages(&j).0 > 0 && usages(&j).1 > 0);
        assert!(
            doublons(&j).is_empty(),
            "doublement, graine {graine} : couple réemployé {:?}",
            doublons(&j)[0]
        );
    }
    // la graine est un champ du flux : aucun couple commun à deux graines (balayages)
    assert!(
        doublons(&union).is_empty(),
        "balayages de graines différentes : couple commun"
    );
    // debit (graine fixe) : pas préparatoire + 3 pas
    let j = executer(&[
        "intrication",
        "debit",
        "--l",
        "2",
        "--t",
        "2",
        "--pas",
        "3",
        "--or",
        "0",
    ]);
    assert!(usages(&j).0 > 0 && usages(&j).1 > 0);
    assert!(
        doublons(&j).is_empty(),
        "debit : couple réemployé {:?}",
        doublons(&j)[0]
    );
    // portes : les seuls couples réemployés sont ceux des réemplois volontaires nommés ; les portes (e) et (e2), en mode à
    // domaines (graines 31 et 37), n'en ont aucun
    let s = sortie("portes");
    let j = executer(&["intrication", "portes", "--rapide", "--sortie", s.as_str()]);
    let permis: Vec<(u32, u32)> = REEMPLOIS_PORTES
        .iter()
        .map(|(g, _)| ((g & 0xFFFF_FFFF) as u32, (g >> 32) as u32))
        .collect();
    for d in doublons(&j) {
        assert!(
            permis.contains(&(d[0], d[1])),
            "portes : réemploi NON nommé, clé ({}, {}) : {d:?}",
            d[0],
            d[1]
        );
    }
    for g in [31u32, 37] {
        assert!(
            j.iter().any(|c| (c[0], c[1]) == (g, 0)),
            "portes : flux de la graine {g} absent du journal"
        );
        assert!(
            doublons(&j).iter().all(|d| (d[0], d[1]) != (g, 0)),
            "portes (e)/(e2), graine {g} : couple réemployé"
        );
    }
    let _ = std::fs::remove_dir_all(&base);
}
