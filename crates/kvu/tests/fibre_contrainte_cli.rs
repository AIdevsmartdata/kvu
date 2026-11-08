//! Contrats S4 : sorties, déterminisme, fils, rejeu et erreurs de la CLI réelle.
use kvu::commun::ChampHote;
use kvu::fibre_contrainte::alea::{Flux, TAG_DEPART};
use kvu::rg::bloc::{bloquer, exp_q, OptionsBloc};
use kvu::rg::geometrie::Geometrie;
use kvu::stockage::npy::{ecrire_champ, FormatChamp};
use kvu::stockage::GardeSsd;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Scratch(PathBuf);
impl Scratch {
    fn nouveau(nom: &str) -> Self {
        static N: AtomicUsize = AtomicUsize::new(0);
        let base = std::env::var_os("KVU_SCRATCH")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/scratch_tests")
            });
        let p = base.join(format!(
            "fibre_contrainte_{nom}_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
    fn chemin(&self, nom: &str) -> PathBuf {
        self.0.join(nom)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn kvu(args: &[String]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_kvu"))
        .args(args)
        .output()
        .unwrap()
}
fn succes(o: &Output) {
    assert_eq!(
        o.status.code(),
        Some(0),
        "stdout : {}\nstderr : {}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
}
fn run(args: &[String], sortie: &Path) {
    let mut a = vec!["fibre".into(), "echantillonner".into()];
    a.extend_from_slice(args);
    a.extend(["--sortie".into(), sortie.to_string_lossy().into_owned()]);
    succes(&kvu(&a));
}
fn mots(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}
fn doc(p: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}
fn verifier(sortie: &Path, n: usize, jouet: bool) -> Value {
    let csv = std::fs::read_to_string(sortie.join("echantillons.csv")).unwrap();
    assert_eq!(csv.lines().count(), n + 1);
    assert_eq!(
        csv.lines().next().unwrap(),
        if jouet {
            "chaine,mesure,obs1,obs2,accepte,delta_h,residu_max,echec,nb_newton"
        } else {
            "chaine,mesure,P,S,penalite,accepte,delta_h,residu_max,echec,nb_newton"
        }
    );
    let colonnes = if jouet { 9 } else { 10 };
    for l in csv.lines().skip(1) {
        assert_eq!(l.split(',').count(), colonnes);
    }
    let m = doc(&sortie.join("manifest_regen.json"));
    assert!(kvu::regen::schema::valider_manifeste(&m).is_empty(), "{m}");
    assert!(m["duree_s"].is_null());
    assert_eq!(m["fichiers_hachage"], "disque");
    assert!(!m["commande"]
        .as_array()
        .unwrap()
        .iter()
        .any(|x| x == "--sortie" || x == "--outdir"));
    for f in m["fichiers"].as_array().unwrap() {
        let p = sortie.join(f["nom"].as_str().unwrap());
        assert_eq!(
            std::fs::metadata(&p).unwrap().len(),
            f["octets"].as_u64().unwrap()
        );
        assert_eq!(
            kvu::regen::manifeste::sha_fichier(&p).unwrap(),
            f["sha256"].as_str().unwrap()
        );
    }
    assert!(sortie.join("manifest.json").exists() && sortie.join("COMPLETE").exists());
    m
}
fn identiques(a: &Path, b: &Path) {
    let noms = |p: &Path| {
        let mut n: Vec<_> = std::fs::read_dir(p)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .filter(|n| n != "manifest.json" && n != "COMPLETE")
            .collect();
        n.sort();
        n
    };
    assert_eq!(noms(a), noms(b));
    // manifest.json : comparé après retrait des SEULS champs qui dépendent de l'exécution (horodatages, pid, répertoire de sortie, argv/cwd, garde SSD)
    let nettoie = |p: &Path| {
        let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(p.join("manifest.json")).unwrap()).unwrap();
        for k in ["pid", "debut_unix", "fin_unix", "garde_ssd", "argv", "cwd"] {
            v.as_object_mut().unwrap().remove(k);
        }
        v
    };
    assert_eq!(nettoie(a), nettoie(b), "manifest.json hors horodatage, pid, argv, cwd, garde SSD");
    for n in noms(a) {
        assert_eq!(
            std::fs::read(a.join(&n)).unwrap(),
            std::fs::read(b.join(&n)).unwrap(),
            "fichier {n:?}"
        );
    }
}
fn fixture(s: &Scratch) -> Vec<String> {
    let flux = Flux::nouveau(26102612042);
    let q = ChampHote {
        l: 6,
        q: (0..4 * 6_usize.pow(4))
            .flat_map(|i| {
                exp_q(std::array::from_fn(|a| {
                    0.5 * flux.gauss(0, 0, (3 * i + a) as u32, TAG_DEPART)
                }))
            })
            .collect(),
    };
    let v = bloquer(
        &q,
        &OptionsBloc {
            geometrie: Geometrie::TroisCentre,
            fils: 1,
            angle_so3: false,
        },
    )
    .unwrap()
    .grossier;
    for (nom, champ) in [("u0.npy", &q), ("v.npy", &v)] {
        ecrire_champ(
            &s.chemin(nom),
            champ,
            FormatChamp::Q64,
            &GardeSsd::machine(),
        )
        .unwrap();
    }
    vec![
        "--entree-v".into(),
        s.chemin("v.npy").to_string_lossy().into_owned(),
        "--entree-u0".into(),
        s.chemin("u0.npy").to_string_lossy().into_owned(),
    ]
}

#[test]
fn fibre_contrainte_jouets_et_rejeu() {
    let s = Scratch::nouveau("jouets");
    for (i, args) in [
        "--jouet courbe --methode rattle --masse 1,2",
        "--jouet deux-liens --t 0.3 --beta 2.3 --methode penalite --v 0,1,0,0",
        "--jouet deux-liens --t 0.3 --beta 2.3 --methode rattle",
    ]
    .iter()
    .enumerate()
    {
        let mut args = mots(args);
        args.extend(mots(
            "--therm 2 --nmeas 3 --skip 2 --chaines 2 --graine 26102612051 --fils 2",
        ));
        let sortie = s.chemin(&format!("jouet{i}"));
        run(&args, &sortie);
        verifier(&sortie, 6, true);
        if i == 0 {
            let rejeu = s.chemin("rejeu");
            let r = kvu(&[
                "regen".into(),
                sortie
                    .join("manifest_regen.json")
                    .to_string_lossy()
                    .into_owned(),
                "--sortie".into(),
                rejeu.to_string_lossy().into_owned(),
            ]);
            succes(&r);
            assert!(String::from_utf8_lossy(&r.stdout).contains("PASS REGEN"));
            let rapport = doc(&rejeu.join("rapport_regen.json"));
            assert!(rapport["fichiers"]
                .as_array()
                .unwrap()
                .iter()
                .all(|f| f["etat"] == "OK"));
            for nom in ["echantillons.csv", "resume.json"] {
                assert_eq!(
                    std::fs::read(sortie.join(nom)).unwrap(),
                    std::fs::read(rejeu.join("rejeu").join(nom)).unwrap()
                );
            }
            let mut invalide = doc(&sortie.join("manifest_regen.json"));
            invalide.as_object_mut().unwrap().remove("graines");
            std::fs::write(s.chemin("invalide.json"), invalide.to_string()).unwrap();
            assert_ne!(
                kvu(&[
                    "regen".into(),
                    s.chemin("invalide.json").to_string_lossy().into_owned(),
                    "--sortie".into(),
                    s.chemin("refus").to_string_lossy().into_owned()
                ])
                .status
                .code(),
                Some(0)
            );
            assert!(!s.chemin("refus").exists());
        }
    }
}

#[test]
fn fibre_contrainte_fib_det_et_fils() {
    let s = Scratch::nouveau("det_fils");
    let entrees = fixture(&s);
    for methode in ["penalite", "rattle"] {
        let mut args = mots(&format!("--methode {methode} --L 6 --beta 2.3 --nmeas 2 --therm 0 --skip 1 --graine 26102612061 --dump-tous 1 --dt 0.002 --duree 0.004 --kappa 2e-3"));
        args.extend_from_slice(&entrees);
        let reference = s.chemin(&format!("{methode}_reference"));
        let mut un = args.clone();
        un.extend(mots("--fils 1"));
        run(&un, &reference);
        verifier(&reference, 2, false);
        assert_eq!(
            doc(&reference.join("manifest_regen.json"))["entrees"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        let double = s.chemin(&format!("{methode}_double"));
        run(&un, &double);
        identiques(&reference, &double);
        // Trois chaînes en A : le travail des chaînes est réellement réparti.
        if methode == "penalite" {
            args.extend(mots("--chaines 3"));
        }
        let mut premiere: Option<PathBuf> = None;
        for fils in [1, 4, 8] {
            let sortie = s.chemin(&format!("{methode}_fils{fils}"));
            let mut a = args.clone();
            a.extend(mots(&format!("--fils {fils}")));
            run(&a, &sortie);
            verifier(&sortie, if methode == "penalite" { 6 } else { 2 }, false);
            if let Some(p) = &premiere {
                identiques(p, &sortie);
            } else {
                premiere = Some(sortie);
            }
        }
    }
}

#[test]
fn fibre_contrainte_erreurs_usage_et_sortie_neuve() {
    let s = Scratch::nouveau("erreurs");
    ecrire_champ(
        &s.chemin("v.npy"),
        &ChampHote::froid(4).unwrap(),
        FormatChamp::Q64,
        &GardeSsd::machine(),
    )
    .unwrap();
    let v = s.chemin("v.npy").to_string_lossy().into_owned();
    for args in [
        format!("--methode penalite --L 8 --beta 2.3 --entree-v {v}"),
        format!("--methode penalite --L 6 --beta 2.3 --entree-v {v}"),
        "--jouet deux-liens --t 0.3 --beta 2.3 --methode penalite --kappa -1".into(),
        "--jouet courbe --methode rattle --dt 0".into(),
        "--jouet courbe --methode rattle --dt -0.1".into(),
        "--jouet courbe --methode rattle --inconnue 1".into(),
        "--jouet courbe --methode rattle --skip 0".into(),
        "--jouet courbe --methode rattle --masse 0,1".into(),
        "--jouet deux-liens --methode rattle --t 1".into(),
        "--jouet deux-liens --methode rattle --t 0.3 --v 2,0,0,0".into(),
        "--jouet courbe --methode rattle --fils 0".into(),
    ] {
        let mut a = mots("fibre echantillonner");
        a.extend(mots(&args));
        a.extend([
            "--sortie".into(),
            s.chemin("interdite").to_string_lossy().into_owned(),
        ]);
        let o = kvu(&a);
        assert_eq!(
            o.status.code(),
            Some(2),
            "{args} : {}",
            String::from_utf8_lossy(&o.stderr)
        );
        assert!(!s.chemin("interdite").exists());
    }
    let a = mots("--methode penalite --kappa inf --L 2 --beta 2.3 --nmeas 2 --fils 1");
    let sortie = s.chemin("libre");
    run(&a, &sortie);
    verifier(&sortie, 2, false);
    let avant = std::fs::read(sortie.join("echantillons.csv")).unwrap();
    let mut cmd = mots("fibre echantillonner");
    cmd.extend(a);
    cmd.extend(["--sortie".into(), sortie.to_string_lossy().into_owned()]);
    assert_eq!(kvu(&cmd).status.code(), Some(2));
    assert_eq!(
        avant,
        std::fs::read(sortie.join("echantillons.csv")).unwrap()
    );
}

#[test]
fn fibre_contrainte_depart_froid_projection_et_l_impair() {
    let s = Scratch::nouveau("projection");
    // Contrôle des deux branches de continuation (une ou huit étapes).
    for (i, angle) in [1e-5, 0.002].into_iter().enumerate() {
        let v = ChampHote {
            l: 2,
            q: (0..4 * 2_usize.pow(4))
                .flat_map(|_| exp_q([angle, 0.0, 0.0]))
                .collect(),
        };
        let chemin = s.chemin(&format!("v{i}.npy"));
        ecrire_champ(&chemin, &v, FormatChamp::Q64, &GardeSsd::machine()).unwrap();
        let mut args = mots("--methode penalite --L 6 --beta 2.3 --nmeas 0 --fils 1");
        args.extend(["--entree-v".into(), chemin.to_string_lossy().into_owned()]);
        let sortie = s.chemin(&format!("projection{i}"));
        run(&args, &sortie);
        verifier(&sortie, 0, false);
        let fin = kvu::stockage::npy::lire_champ(&sortie.join("etat_final_c000.npy")).unwrap();
        let reseau =
            std::sync::Arc::new(kvu::fibre_contrainte::jacobien::Reseau::nouveau(6).unwrap());
        let cible = (0..reseau.n_c).map(|c| v.lien(c)).collect();
        let modele =
            kvu::fibre_contrainte::modele_su2::FibreSu2::nouveau(reseau, cible, 2.3, 1).unwrap();
        assert!(modele.residu_max(&fin).unwrap() <= 1e-12);
    }
    // L = 9 reste autorisé par S4, même si ChampHote::froid et ChainePenalite
    // sont historiquement réservés aux réseaux pairs.
    let v = ChampHote {
        l: 3,
        q: (0..4 * 3_usize.pow(4))
            .flat_map(|_| [1.0, 0.0, 0.0, 0.0])
            .collect(),
    };
    let chemin = s.chemin("v9.npy");
    ecrire_champ(&chemin, &v, FormatChamp::Q64, &GardeSsd::machine()).unwrap();
    let mut args = mots("--methode penalite --L 9 --beta 2.3 --nmeas 0 --fils 1");
    args.extend(["--entree-v".into(), chemin.to_string_lossy().into_owned()]);
    run(&args, &s.chemin("impair"));
    verifier(&s.chemin("impair"), 0, false);
    let fin = kvu::stockage::npy::lire_champ(&s.chemin("impair/etat_final_c000.npy")).unwrap();
    assert_eq!(fin.l, 9);
    // Cible sur la coupure du logarithme : validation rc 8 avant échantillonnage.
    let v = ChampHote {
        l: 2,
        q: (0..4 * 2_usize.pow(4))
            .flat_map(|_| [-1.0, 0.0, 0.0, 0.0])
            .collect(),
    };
    let chemin = s.chemin("coupure.npy");
    ecrire_champ(&chemin, &v, FormatChamp::Q64, &GardeSsd::machine()).unwrap();
    let mut args =
        mots("fibre echantillonner --methode rattle --L 6 --beta 2.3 --nmeas 1 --fils 1");
    args.extend([
        "--entree-v".into(),
        chemin.to_string_lossy().into_owned(),
        "--sortie".into(),
        s.chemin("refus").to_string_lossy().into_owned(),
    ]);
    let o = kvu(&args);
    assert_eq!(
        o.status.code(),
        Some(8),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(!s.chemin("refus").exists());
}
