#![forbid(unsafe_code)]

use kvu::commun::ChampHote;
use kvu::hess::bases::{Base, Coord};
use kvu::hess::dual::jvp;
use kvu::hess::espaces::{construire, Espace, ModeProj};
use kvu::hess::hamilt::{Hessienne, MutantHess};
use kvu::hess::lobpcg::Op;
use kvu::rg::bloc::{bloquer, OptionsBloc};
use kvu::rg::geometrie::Geometrie;
use kvu::stockage::garde::GardeSsd;
use kvu::stockage::npy::{ecrire_champ, ecrire_f64, lire_champ, lire_f64, FormatChamp};
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

struct Dossier {
    chemin: PathBuf,
    debut: Instant,
    nom: &'static str,
}
impl Dossier {
    fn nouveau(nom: &'static str) -> Self {
        let instant = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("horloge")
            .as_nanos();
        let chemin = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("hess_cli_{nom}_{}_{instant}", std::process::id()));
        std::fs::create_dir_all(chemin.parent().expect("parent temporaire"))
            .expect("parent temporaire");
        std::fs::create_dir(&chemin).expect("dossier temporaire neuf");
        Self {
            chemin,
            debut: Instant::now(),
            nom,
        }
    }
    fn fichier(&self, nom: &str) -> String {
        self.chemin
            .join(nom)
            .to_str()
            .expect("chemin UTF-8")
            .to_owned()
    }
    fn champ(&self, nom: &str, c: &ChampHote, format: FormatChamp) -> String {
        let chemin = self.fichier(nom);
        ecrire_champ(Path::new(&chemin), c, format, &GardeSsd::nouvelle(0.0))
            .expect("écriture du champ");
        chemin
    }
    fn vecteurs(&self, nom: &str, forme: &[usize], v: &[f64]) -> String {
        let chemin = self.fichier(nom);
        ecrire_f64(Path::new(&chemin), forme, v, &GardeSsd::nouvelle(0.0))
            .expect("écriture des vecteurs");
        chemin
    }
}
impl Drop for Dossier {
    fn drop(&mut self) {
        eprintln!("{} : {:.3} s", self.nom, self.debut.elapsed().as_secs_f64());
        std::fs::remove_dir_all(&self.chemin).expect("nettoyage du dossier temporaire");
    }
}

// SplitMix64 et tirage Haar de hess_hv.rs ; aucune dépendance supplémentaire.
struct Aleatoire(u64);
impl Aleatoire {
    fn entier(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn reel(&mut self) -> f64 {
        2.0 * ((self.entier() >> 11) as f64 / 9_007_199_254_740_992.0) - 1.0
    }
    fn quaternion(&mut self) -> [f64; 4] {
        loop {
            let q = [self.reel(), self.reel(), self.reel(), self.reel()];
            let n2: f64 = q.iter().map(|x| x * x).sum();
            if n2 > 0.0 && n2 <= 1.0 {
                return q.map(|x| x / n2.sqrt());
            }
        }
    }
}
fn configuration(l: usize, graine: u64) -> ChampHote {
    let mut alea = Aleatoire(graine);
    ChampHote {
        l,
        q: (0..4 * l.pow(4)).flat_map(|_| alea.quaternion()).collect(),
    }
}
fn froid(l: usize) -> ChampHote {
    ChampHote {
        l,
        q: (0..4 * l.pow(4))
            .flat_map(|_| [1.0, 0.0, 0.0, 0.0])
            .collect(),
    }
}
fn vecteur(n: usize, graine: u64) -> Vec<f64> {
    let mut alea = Aleatoire(graine);
    (0..n).map(|_| alea.reel()).collect()
}
fn pool(fils: usize) -> Arc<ThreadPool> {
    Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(fils)
            .build()
            .expect("création du pool"),
    )
}
fn kvu(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_kvu"))
        .args(args)
        .output()
        .expect("lancement de kvu")
}
fn succes(args: &[&str]) {
    let o = kvu(args);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{args:?} : {}",
        String::from_utf8_lossy(&o.stderr)
    );
}
fn bits(x: &[f64], y: &[f64]) {
    assert_eq!(x.len(), y.len(), "longueurs des vecteurs");
    for (i, (a, b)) in x.iter().zip(y).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "composante {i}");
    }
}
fn json(chemin: &str) -> Value {
    serde_json::from_slice(&std::fs::read(chemin).expect("lecture JSON")).expect("JSON valide")
}

#[test]
fn hv_cli_egal_bibliotheque() {
    let d = Dossier::nouveau("hv_cli_egal_bibliotheque");
    let c = configuration(2, 201);
    let n = 12 * c.v();
    for (format, nom) in [(FormatChamp::C16, "c16.npy"), (FormatChamp::Q64, "q64.npy")] {
        let chemin = d.champ(nom, &c, format);
        let relu = lire_champ(Path::new(&chemin)).expect("champ relu");
        for (base, coord) in [
            (Base::S2, Coord::Additive),
            (Base::S2, Coord::Gauche),
            (Base::Ym4, Coord::Gauche),
        ] {
            let h = Hessienne::nouveau(&relu, pool(3), base, coord, 2.3, MutantHess::Aucun)
                .expect("Hessienne");
            for forme in [vec![n], vec![3, n]] {
                let v = vecteur(forme.iter().product(), 202);
                let entree = d.vecteurs("v.npy", &forme, &v);
                let sortie = d.fichier("y.npy");
                succes(&[
                    "hess",
                    "hv",
                    "--config",
                    &chemin,
                    "--vecteurs",
                    &entree,
                    "--sortie",
                    &sortie,
                    "--base",
                    base.nom(),
                    "--coord",
                    coord.nom(),
                    "--beta",
                    "2.3",
                    "--fils",
                    "3",
                ]);
                let (forme_lue, y) = lire_f64(Path::new(&sortie)).expect("sortie HV");
                assert_eq!(forme_lue, forme, "forme conservée");
                let mut attendu = vec![f64::NAN; v.len()];
                for (x, y) in v.chunks(n).zip(attendu.chunks_mut(n)) {
                    h.hv(x, y);
                }
                bits(&y, &attendu);
            }
        }
    }
}

#[test]
fn jvp_cli() {
    let d = Dossier::nouveau("jvp_cli");
    let c = configuration(3, 211);
    let config = d.champ("c.npy", &c, FormatChamp::Q64);
    let n = 12 * c.v();
    for forme in [vec![n], vec![2, n]] {
        let v = vecteur(forme.iter().product(), 212);
        let entree = d.vecteurs("v.npy", &forme, &v);
        let sortie = d.fichier("y.npy");
        let ubar = d.fichier("ubar.npy");
        succes(&[
            "hess",
            "jvp",
            "--config",
            &config,
            "--vecteurs",
            &entree,
            "--sortie",
            &sortie,
            "--ubar-sortie",
            &ubar,
            "--fils",
            "2",
        ]);
        let (forme_lue, y) = lire_f64(Path::new(&sortie)).expect("sortie JVP");
        assert_eq!(
            forme_lue,
            if forme.len() == 1 {
                vec![12]
            } else {
                vec![2, 12]
            },
            "forme grossière"
        );
        let p = pool(2);
        let attendu: Vec<_> = v
            .chunks(n)
            .flat_map(|v| jvp(&p, &c, v).expect("JVP bibliothèque").dubar_v)
            .collect();
        bits(&y, &attendu);
        let u = lire_champ(Path::new(&ubar)).expect("champ grossier lisible");
        let b = bloquer(
            &c,
            &OptionsBloc {
                geometrie: Geometrie::TroisCentre,
                fils: 2,
                angle_so3: false,
            },
        )
        .expect("blocage de référence")
        .grossier;
        assert_eq!(u.l, b.l, "taille grossière");
        assert_eq!(u.q.len(), b.q.len(), "longueur grossière");
        for (i, (x, y)) in u.q.iter().zip(&b.q).enumerate() {
            assert!((x - y).abs() <= 1e-14, "Ū composante {i} : {x} / {y}");
        }
    }
}

#[test]
fn lmin_cli_json() {
    let d = Dossier::nouveau("lmin_cli_json");
    let c = froid(4);
    let config = d.champ("c.npy", &c, FormatChamp::C16);
    let rapport = d.fichier("r.json");
    let sortie = d.fichier("v.npy");
    succes(&[
        "hess",
        "lmin",
        "--config",
        &config,
        "--espace",
        "libre",
        "--projecteur",
        "cg",
        "--sortie",
        &rapport,
        "--vecteur-sortie",
        &sortie,
    ]);
    let j = json(&rapport);
    for cle in [
        "lambda_min",
        "eps_prime",
        "dim",
        "residu_projete_rel",
        "solveur",
        "projecteur_apres",
    ] {
        assert!(j.get(cle).is_some(), "clé JSON {cle} absente");
    }
    assert!(
        j["solveur"]["iterations"].as_u64().is_some(),
        "nombre d'itérations"
    );
    let lambda = j["lambda_min"].as_f64().expect("λ_min numérique");
    let eps = j["eps_prime"].as_f64().expect("ε′ numérique");
    // La lecture décimale du JSON peut déplacer λ d'un ulp.
    let ecart = (eps - (1.0 - lambda)).abs();
    assert!(
        ecart <= f64::EPSILON * lambda.abs().max(1.0),
        "ε′ = 1 − λ_min : écart après lecture JSON {ecart:e}"
    );
    let (forme, v) = lire_f64(Path::new(&sortie)).expect("vecteur propre écrit");
    assert_eq!(forme, vec![12 * c.v()], "forme du vecteur propre");
    assert_eq!(v.len(), 12 * c.v(), "longueur du vecteur propre");
    assert!(v.iter().all(|x| x.is_finite()), "vecteur propre fini");
}

#[test]
fn espace_cli() {
    let d = Dossier::nouveau("espace_cli");
    let c = configuration(2, 221);
    let config = d.champ("c.npy", &c, FormatChamp::Q64);
    let n = 12 * c.v();
    let z = vecteur(n, 222);
    let entree = d.vecteurs("z.npy", &[n], &z);
    let sortie = d.fichier("p.npy");
    let rapport = d.fichier("r.json");
    succes(&[
        "hess",
        "espace",
        "--config",
        &config,
        "--espace",
        "kerB",
        "--projecteur",
        "svd",
        "--projeter",
        &entree,
        "--sortie",
        &sortie,
        "--rapport",
        &rapport,
    ]);
    let e = construire(Espace::KerB, &c, pool(4), ModeProj::Svd).expect("espace bibliothèque");
    assert_eq!(
        json(&rapport)["dim"],
        serde_json::to_value(e.dim).expect("dimension JSON"),
        "dimension de l'espace"
    );
    let (forme, y) = lire_f64(Path::new(&sortie)).expect("projection écrite");
    assert_eq!(forme, vec![n], "forme de la projection");
    let mut attendu = vec![f64::NAN; n];
    e.proj.appliquer(&z, &mut attendu);
    bits(&y, &attendu);
}

#[test]
fn cli_erreurs() {
    let d = Dossier::nouveau("cli_erreurs");
    let c = froid(2);
    let n = 12 * c.v();
    let config = d.champ("c.npy", &c, FormatChamp::Q64);
    let v = d.vecteurs("v.npy", &[n], &vecteur(n, 231));
    let mauvais = d.vecteurs("forme.npy", &[2, n / 2], &vecteur(n, 232));
    let sortie = d.fichier("interdit.npy");
    let rapport = d.fichier("interdit.json");
    let cas = [
        (
            vec![
                "hess",
                "espace",
                "--config",
                &config,
                "--espace",
                "inconnu",
                "--projeter",
                &v,
                "--sortie",
                &sortie,
                "--rapport",
                &rapport,
            ],
            "espace « inconnu » inconnu",
        ),
        (
            vec![
                "hess",
                "hv",
                "--config",
                &config,
                "--vecteurs",
                &v,
                "--sortie",
                &sortie,
                "--base",
                "inconnue",
            ],
            "base « inconnue » inconnue",
        ),
        (
            vec![
                "hess",
                "hv",
                "--config",
                &config,
                "--vecteurs",
                &v,
                "--sortie",
                &sortie,
                "--base",
                "ym4",
                "--coord",
                "additive",
            ],
            "les coordonnées additives sont définies en base s2",
        ),
        (
            vec![
                "hess",
                "hv",
                "--config",
                &config,
                "--vecteurs",
                &v,
                "--sortie",
                &sortie,
                "--fils",
                "0",
            ],
            "--fils doit être ≥ 1",
        ),
        (
            vec![
                "hess",
                "hv",
                "--config",
                &config,
                "--vecteurs",
                &mauvais,
                "--sortie",
                &sortie,
            ],
            "attendu (k, 192) ou (192,)",
        ),
    ];
    for (args, message) in cas {
        let o = kvu(&args);
        assert_eq!(o.status.code(), Some(2), "usage invalide {args:?}");
        let erreur = String::from_utf8_lossy(&o.stderr);
        assert!(
            erreur.contains(message),
            "message français attendu {message:?}, reçu {erreur:?}"
        );
        assert!(
            !Path::new(&sortie).exists() && !Path::new(&rapport).exists(),
            "aucune sortie après erreur"
        );
    }
}

#[test]
fn sortie_independante_des_fils() {
    let d = Dossier::nouveau("sortie_independante_des_fils");
    for commande in ["hv", "jvp", "lmin"] {
        let c = if commande == "jvp" {
            configuration(3, 241)
        } else {
            configuration(4, 242)
        };
        let config = d.champ("c.npy", &c, FormatChamp::Q64);
        let n = 12 * c.v();
        let v = d.vecteurs("v.npy", &[n], &vecteur(n, 243));
        let mut fichiers = Vec::new();
        for fils in ["1", "8"] {
            let sortie = d.fichier(&format!("{commande}_{fils}.sortie"));
            let auxiliaire = d.fichier(&format!("{commande}_{fils}.npy"));
            let mut args = vec![
                "hess", commande, "--config", &config, "--sortie", &sortie, "--fils", fils,
            ];
            match commande {
                "hv" => args.extend([
                    "--vecteurs",
                    &v,
                    "--base",
                    "s2",
                    "--coord",
                    "additive",
                    "--beta",
                    "2.3",
                ]),
                "jvp" => args.extend(["--vecteurs", &v, "--ubar-sortie", &auxiliaire]),
                "lmin" => args.extend([
                    "--espace",
                    "libre",
                    "--projecteur",
                    "cg",
                    "--vecteur-sortie",
                    &auxiliaire,
                ]),
                _ => unreachable!("commande de test connue"),
            }
            succes(&args);
            let principal = std::fs::read(&sortie).expect("sortie principale");
            let annexe = if commande == "hv" {
                Vec::new()
            } else {
                std::fs::read(&auxiliaire).expect("sortie annexe")
            };
            fichiers.push((principal, annexe));
        }
        assert_eq!(
            fichiers[0], fichiers[1],
            "{commande} : identité des fichiers à l'octet selon les fils"
        );
    }
}

#[test]
fn aide_et_sous_commandes() {
    let _d = Dossier::nouveau("aide_et_sous_commandes");
    let aide = kvu(&["--help"]);
    assert!(aide.status.success(), "aide générale");
    assert!(
        String::from_utf8_lossy(&aide.stdout)
            .split_whitespace()
            .any(|mot| mot == "hess"),
        "hess dans l'aide générale"
    );
    let aide = kvu(&["hess", "--help"]);
    assert!(aide.status.success(), "aide de hess");
    let texte = String::from_utf8_lossy(&aide.stdout);
    for commande in ["hv", "jvp", "espace", "lmin"] {
        assert!(
            texte.split_whitespace().any(|mot| mot == commande),
            "{commande} dans l'aide"
        );
    }
    assert!(
        kvu::cli::SOUS_COMMANDES.contains(&"hess"),
        "hess enregistré dans SOUS_COMMANDES"
    );
}
