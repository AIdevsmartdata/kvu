//! Porte G-CHEMINS (O2 de la relecture du 11/11) : la géométrie d'une configuration ne dépend pas de la façon d'écrire son
//! chemin (absolu, relatif avec ou sans `./`, fichier nu depuis son dossier, `..`, détour `c0/..`, lien symbolique placé
//! ailleurs).
//! UN SEUL test dans ce binaire : il change le répertoire courant du processus (`set_current_dir`), état global qu'aucun
//! autre test ne doit partager. Mutant M8 : dossiers pris dans le chemin tel qu'écrit ⇒ la porte doit le voir.

#[path = "donnees/stockage/support.rs"]
mod support;

use kvu::commun::ChampHote;
use kvu::geom::{
    ecrire_geometrie, examiner_voisinage_avec, geometrie_voisine, ChampGeo, Chemin, GeoSpec,
    MutantVoisinage, Torsion,
};
use kvu::stockage::garde::GardeSsd;
use kvu::stockage::npy::{self, FormatChamp};
use std::fs;
use std::path::{Path, PathBuf};
use support::Scratch;

/// Lot `d/` : configuration cubique L = 4 dans `d/c0/champ.npy`, lien symbolique `d/ailleurs/lien.npy` vers elle.
fn lot(sc: &Scratch, nom: &str) -> PathBuf {
    let d = sc.p(nom);
    fs::create_dir_all(d.join("c0")).unwrap();
    fs::create_dir_all(d.join("ailleurs")).unwrap();
    let mut q = Vec::with_capacity(16 * 256);
    for li in 0..4 * 256 {
        let t = 1e-3 * (li + 1) as f64;
        q.extend_from_slice(&[t.cos(), 0.0, 0.0, t.sin()]);
    }
    let c = ChampGeo::depuis_hote(ChampHote { l: 4, q }, 3).unwrap();
    let cfg = d.join("c0").join("champ.npy");
    npy::ecrire_champ_geo(&cfg, &c, FormatChamp::C16, &GardeSsd::nouvelle(0.0)).unwrap();
    std::os::unix::fs::symlink(&cfg, d.join("ailleurs").join("lien.npy")).unwrap();
    d
}

/// (répertoire courant, chemin écrit) : six écritures du même fichier.
fn cas(d: &Path) -> Vec<(PathBuf, String)> {
    let abs = d.join("c0").join("champ.npy").display().to_string();
    vec![
        (d.to_path_buf(), "c0/champ.npy".into()),
        (d.to_path_buf(), "./c0/champ.npy".into()),
        (d.join("c0"), "champ.npy".into()),
        (d.join("c0"), "../c0/champ.npy".into()),
        (d.join("ailleurs"), "lien.npy".into()),
        (d.join("ailleurs"), abs),
    ]
}

#[test]
fn g_chemins_independance_et_mutant_m8() {
    let sc = Scratch::nouveau("geom_chemins");
    let origine = std::env::current_dir().unwrap();
    let tordue = GeoSpec::cubique(4, 3).avec_torsion(Torsion::analyser("01").unwrap());
    let mut n = 0usize;
    let mut ecarts = 0usize;

    // lot A : geometrie.json tordue à la racine du lot ⇒ la configuration est C1 tordue, quel que soit le chemin
    let a = lot(&sc, "a");
    ecrire_geometrie(&a, &tordue).unwrap();
    // lot B : torsion historique à la racine, sans geometrie.json ⇒ refus code 2, quel que soit le chemin
    let b = lot(&sc, "b");
    fs::write(b.join("torsion.json"), br#"{"plans": ["01"]}"#).unwrap();
    // lot C : aucune métadonnée ⇒ C0 avec le défaut passé par l'appelant, quel que soit le chemin
    let c = lot(&sc, "c");

    for (cwd, chemin) in cas(&a) {
        std::env::set_current_dir(&cwd).unwrap();
        let lu = npy::lire_champ_geo(Path::new(&chemin), 0);
        n += 1;
        if !lu
            .as_ref()
            .is_ok_and(|ch| ch.geo == tordue && ch.geo.chemin() == Chemin::C1)
        {
            ecarts += 1;
            eprintln!(
                "lot A : cwd={} chemin={chemin} : {:?}",
                cwd.display(),
                lu.map(|ch| ch.geo)
            );
        }
    }
    for (cwd, chemin) in cas(&b) {
        std::env::set_current_dir(&cwd).unwrap();
        n += 1;
        let code = geometrie_voisine(Path::new(&chemin))
            .err()
            .map(|e| e.code_sortie());
        if code != Some(2) {
            ecarts += 1;
            eprintln!(
                "lot B : cwd={} chemin={chemin} : code {code:?}",
                cwd.display()
            );
        }
    }
    for (cwd, chemin) in cas(&c) {
        std::env::set_current_dir(&cwd).unwrap();
        n += 1;
        let lu = npy::lire_champ_geo(Path::new(&chemin), 0);
        if !lu.as_ref().is_ok_and(|ch| ch.geo == GeoSpec::cubique(4, 0)) {
            ecarts += 1;
            eprintln!(
                "lot C : cwd={} chemin={chemin} : {:?}",
                cwd.display(),
                lu.map(|ch| ch.geo)
            );
        }
    }

    // lot D (relecture c04b, O2) : configuration C0 à la RACINE du lot, géométrie tordue d'une AUTRE configuration dans c0/ :
    // « champ.npy », « ./champ.npy », « c0/../champ.npy » et le chemin absolu donnent tous C0 (le détour par c0/.. ne
    // doit pas faire voir c0/geometrie.json)
    let dd = sc.p("d");
    fs::create_dir_all(dd.join("c0")).unwrap();
    let mut q = Vec::with_capacity(16 * 256);
    for _ in 0..4 * 256 {
        q.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
    }
    let c0 = ChampGeo::depuis_hote(ChampHote { l: 4, q }, 3).unwrap();
    npy::ecrire_champ_geo(
        &dd.join("champ.npy"),
        &c0,
        FormatChamp::Q64,
        &GardeSsd::nouvelle(0.0),
    )
    .unwrap();
    ecrire_geometrie(&dd.join("c0"), &tordue).unwrap();
    let abs = dd.join("champ.npy").display().to_string();
    for chemin in ["champ.npy", "./champ.npy", "c0/../champ.npy", abs.as_str()] {
        std::env::set_current_dir(&dd).unwrap();
        n += 1;
        let lu = npy::lire_champ_geo(Path::new(chemin), 0);
        if !lu.as_ref().is_ok_and(|ch| ch.geo == GeoSpec::cubique(4, 0)) {
            ecarts += 1;
            eprintln!("lot D : chemin={chemin} : {:?}", lu.map(|ch| ch.geo));
        }
    }

    // mutant M8 : chemin tel qu'écrit ; depuis la racine du lot A, « c0/champ.npy » ne voit pas la géométrie du lot
    std::env::set_current_dir(&a).unwrap();
    let mutant =
        examiner_voisinage_avec(Path::new("c0/champ.npy"), MutantVoisinage::CheminTelQuEcrit)
            .unwrap()
            .spec
            .map(|(g, _)| g);
    std::env::set_current_dir(&origine).unwrap();

    assert_eq!(n, 22, "comparaisons");
    assert_eq!(ecarts, 0, "G-CHEMINS");
    println!("PASS G-CHEMINS {n}");
    assert_ne!(mutant, Some(tordue), "M8 : chemin tel qu'écrit non détecté");
    println!("PASS M8-CHEMIN-TEL-QU-ECRIT 1 (mutant ROUGE : 1 écart, géométrie du lot non vue : {mutant:?})");
}
