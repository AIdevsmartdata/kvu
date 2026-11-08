// SPDX-License-Identifier: Apache-2.0
//! Compression zstd sans fil auxiliaire (correctif du code 104 sous un plafond de tâches).
//!
//! Adapté, pour la suite générale, du test synthétique du paquet de correctif `kvu_compression_8t` (04/11/2026) : données
//! entièrement synthétiques, aucun ensemble ni GPU. Le paquet d'origine prouvait en plus, sous `strace` avec refus de
//! `clone`/`clone3`, que les trois appels (compression, test, lecture) ne créent aucun fil ; cette preuve demande ptrace et
//! reste hors de la suite. Ici on vérifie le comportement fonctionnel : aller-retour exact, somme de contrôle conservée,
//! lecture d'un cadre `-T4` historique, refus des fichiers corrompus ou tronqués, refus d'écraser, original intact.
//! Le test est ignoré proprement si aucun `zstd` n'est installé (la compression est facultative).

use kvu::commun::ChampHote;
use kvu::heritage::manifest::sha256_hex;
use kvu::stockage::{
    compression,
    garde::GardeSsd,
    npy::{self, FormatChamp},
};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

fn dossier(nom: &str) -> PathBuf {
    let base = std::env::var_os("KVU_SCRATCH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")));
    let d = base.join(format!("compression_sans_fil_{nom}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// Lecture par le zstd système, sans passer par le module (compatibilité du format produit).
fn lecture_systeme(p: &Path) -> Vec<u8> {
    let o = Command::new("zstd").args(["-dc", "-q", "--"]).arg(p).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    o.stdout
}

fn aller_retour(racine: &Path, nom: &str, octets: &[u8]) -> PathBuf {
    let source = racine.join(nom);
    fs::write(&source, octets).unwrap();
    let compresse = compression::compresser(&source).unwrap();
    assert!(!source.exists(), "l'original doit être supprimé après validation");
    assert!(!PathBuf::from(format!("{}.part", compresse.display())).exists());
    compression::tester(&compresse).unwrap();
    assert_eq!(compression::decompresser_en_memoire(&compresse).unwrap(), octets);
    assert_eq!(lecture_systeme(&compresse), octets, "lecture par le zstd système");
    let cadre = fs::read(&compresse).unwrap();
    assert_eq!(&cadre[..4], &[0x28, 0xb5, 0x2f, 0xfd], "nombre magique zstd");
    assert_ne!(cadre[4] & 4, 0, "la somme de contrôle du contenu doit rester active");
    let mut flux = compression::ouvrir_flux(&compresse).unwrap();
    let mut relu = vec![];
    let mut tampon = [0u8; 65521];
    loop {
        let n = flux.read(&mut tampon).unwrap();
        if n == 0 {
            break;
        }
        relu.extend_from_slice(&tampon[..n]);
    }
    flux.terminer().unwrap();
    assert_eq!(relu, octets);
    println!("SHA256 {nom} {} {}", octets.len(), sha256_hex(octets));
    compresse
}

#[test]
fn compression_sans_fil_aller_retour_et_refus() {
    if !compression::disponible() {
        eprintln!("SKIP : aucun zstd dans le PATH (compression facultative)");
        return;
    }
    let opts = compression::options_sans_fil();
    assert_eq!(opts.first(), Some(&"--single-thread"), "options sondées : {opts:?}");
    let racine = dossier("principal");

    aller_retour(&racine, "vide.bin", &[]);
    aller_retour(&racine, "petit.bin", &(0u8..=255).collect::<Vec<_>>());
    let mut etat = 0x6a09e667f3bcc909u64;
    let octets: Vec<u8> = (0..(8 << 20))
        .map(|i| {
            etat ^= etat << 13;
            etat ^= etat >> 7;
            etat ^= etat << 17;
            if i % 4 == 0 {
                0
            } else {
                etat as u8
            }
        })
        .collect();
    let compresse = aller_retour(&racine, "mixte_8Mio.bin", &octets);

    // Un cadre « -T4 » produit à l'ancienne reste lisible par le module corrigé
    // (E/S synchrones pour ce témoin seulement, quand le zstd installé connaît l'option).
    let ancien_source = racine.join("ancien.bin");
    let ancien = racine.join("ancien.bin.zst");
    fs::write(&ancien_source, &octets).unwrap();
    let mut cmd = Command::new("zstd");
    cmd.args(["-3", "-T4"]);
    if opts.contains(&"--no-asyncio") {
        cmd.arg("--no-asyncio");
    }
    let r = cmd.args(["-q", "-o"]).arg(&ancien).arg("--").arg(&ancien_source).output().unwrap();
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    compression::tester(&ancien).unwrap();
    assert_eq!(compression::decompresser_en_memoire(&ancien).unwrap(), octets);

    // Champs identité (aucun pas de Monte-Carlo) : en-têtes, forme et valeurs survivent à la compression.
    let champ = ChampHote::froid(4).unwrap();
    for (nom, format) in [("champ_q64.npy", FormatChamp::Q64), ("champ_c16.npy", FormatChamp::C16)] {
        let p = racine.join(nom);
        npy::ecrire_champ(&p, &champ, format, &GardeSsd::nouvelle(1.0)).unwrap();
        let original = fs::read(&p).unwrap();
        assert_eq!(&original[..6], b"\x93NUMPY");
        fs::remove_file(&p).unwrap();
        let zippe = aller_retour(&racine, nom, &original);
        assert_eq!(npy::inspecter_champ(&zippe).unwrap(), (format, 4));
        assert_eq!(npy::lire_champ(&zippe).unwrap(), champ);
    }

    // Somme de contrôle seule altérée, puis fichier tronqué : refusés au test et à la lecture.
    let cadre = fs::read(&compresse).unwrap();
    let corrompu = racine.join("somme_alteree.zst");
    let mut mauvais = cadre.clone();
    let dernier = mauvais.len() - 1;
    mauvais[dernier] ^= 1;
    fs::write(&corrompu, &mauvais).unwrap();
    assert!(compression::tester(&corrompu).is_err());
    assert!(compression::decompresser_en_memoire(&corrompu).is_err());
    let tronque = racine.join("tronque.zst");
    fs::write(&tronque, &cadre[..cadre.len() - 2]).unwrap();
    assert!(compression::tester(&tronque).is_err());
    assert!(compression::decompresser_en_memoire(&tronque).is_err());

    // Destination existante : refus, source et destination inchangées.
    let existant = racine.join("mixte_8Mio.bin");
    fs::write(&existant, &octets).unwrap();
    assert!(compression::compresser(&existant).is_err());
    assert_eq!(fs::read(&existant).unwrap(), octets);
    assert_eq!(fs::read(&compresse).unwrap(), cadre);

    let _ = fs::remove_dir_all(&racine);
}
