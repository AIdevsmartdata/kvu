//! Tests CPU du cache disque des codes objets (T2, gate RT-2 : miss, hit, poison, plafond LRU).
//! Chaque test travaille dans son propre répertoire sous CARGO_TARGET_TMPDIR (jamais dans /tmp, qui est en RAM).

use kvu::gpu::cache::{self, Cache, MetaEntree};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

fn repertoire_de_test(nom: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("gpu_cache")
        .join(format!("{nom}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn meta(nom: &str, arch: &str) -> MetaEntree {
    MetaEntree {
        format: 0,
        nom: nom.into(),
        arch: arch.into(),
        cle: String::new(),
        version_compilateur: "essai".into(),
        arguments: vec!["-O3".into()],
        sha256_source: "00".into(),
        sha256_code: String::new(),
        taille_code: 0,
    }
}

/// Clé distincte pour chaque numéro d'entrée.
fn cle_n(n: usize) -> String {
    cache::cle(&format!("source {n}"), &["-O3".to_string()], "gfx1151", "essai")
}

fn code_n(n: usize, taille: usize) -> Vec<u8> {
    (0..taille).map(|i| (i * 31 + n * 7) as u8).collect()
}

fn dater(p: &Path, il_y_a_secondes: u64) {
    let f = fs::OpenOptions::new().write(true).open(p).unwrap();
    f.set_modified(SystemTime::now() - Duration::from_secs(il_y_a_secondes))
        .unwrap();
}

fn fichiers(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    v.sort();
    v
}

#[test]
fn manque_puis_ecriture_puis_succes() {
    let d = repertoire_de_test("hit");
    let c = Cache::nouveau(&d, 1 << 20);
    let (k, code) = (cle_n(0), code_n(0, 4096));
    assert!(c.lire("su2", "gfx1151", &k, "hsaco").is_none(), "manque initial");
    c.ecrire("su2", "gfx1151", &k, "hsaco", &code, meta("su2", "gfx1151"))
        .unwrap();
    let e = c.lire("su2", "gfx1151", &k, "hsaco").expect("succès");
    assert_eq!(e.code, code);
    assert_eq!(e.meta.cle, k);
    assert_eq!(e.meta.taille_code, 4096);
    assert_eq!(e.meta.sha256_code, kvu::heritage::manifest::sha256_hex(&code));
    assert_eq!(e.meta.version_compilateur, "essai");
    // Noms de fichiers : <nom>-<arch>-<sha16>.{hsaco,json}, aucun temporaire.
    let radical = format!("su2-gfx1151-{}", &k[..16]);
    assert_eq!(fichiers(&d), [format!("{radical}.hsaco"), format!("{radical}.json")]);
}

#[test]
fn extension_cubin_et_arches_distinctes() {
    let d = repertoire_de_test("cubin");
    let c = Cache::nouveau(&d, 1 << 20);
    let k = cle_n(1);
    c.ecrire("cond", "sm_120", &k, "cubin", &code_n(1, 512), meta("cond", "sm_120"))
        .unwrap();
    assert!(c.lire("cond", "sm_120", &k, "cubin").is_some());
    assert!(
        c.lire("cond", "sm_89", &k, "cubin").is_none(),
        "autre architecture ⇒ autre entrée"
    );
    assert!(c
        .chemin_code("cond", "sm_120", &k, "cubin")
        .to_string_lossy()
        .ends_with(".cubin"));
}

#[test]
fn poison_octets_modifies_evince() {
    let d = repertoire_de_test("poison");
    let c = Cache::nouveau(&d, 1 << 20);
    let (k, code) = (cle_n(2), code_n(2, 2048));
    c.ecrire("su2", "gfx1151", &k, "hsaco", &code, meta("su2", "gfx1151"))
        .unwrap();
    let p = c.chemin_code("su2", "gfx1151", &k, "hsaco");
    let mut empoisonne = code.clone();
    empoisonne[1000] ^= 0xFF; // même taille, un octet changé
    fs::write(&p, &empoisonne).unwrap();
    assert!(
        c.lire("su2", "gfx1151", &k, "hsaco").is_none(),
        "sha256 différent ⇒ évincé"
    );
    assert!(fichiers(&d).is_empty(), "code et json supprimés : {:?}", fichiers(&d));
    // L'entrée peut être réécrite puis relue : « puis recompilée ».
    c.ecrire("su2", "gfx1151", &k, "hsaco", &code, meta("su2", "gfx1151"))
        .unwrap();
    assert_eq!(c.lire("su2", "gfx1151", &k, "hsaco").unwrap().code, code);
}

#[test]
fn poison_taille_nulle_evince() {
    let d = repertoire_de_test("nul");
    let c = Cache::nouveau(&d, 1 << 20);
    let (k, code) = (cle_n(3), code_n(3, 1024));
    c.ecrire("su2", "gfx1151", &k, "hsaco", &code, meta("su2", "gfx1151"))
        .unwrap();
    fs::write(c.chemin_code("su2", "gfx1151", &k, "hsaco"), b"").unwrap();
    assert!(c.lire("su2", "gfx1151", &k, "hsaco").is_none());
    assert!(fichiers(&d).is_empty());
}

#[test]
fn poison_tronque_evince() {
    let d = repertoire_de_test("tronque");
    let c = Cache::nouveau(&d, 1 << 20);
    let (k, code) = (cle_n(4), code_n(4, 1024));
    c.ecrire("su2", "gfx1151", &k, "hsaco", &code, meta("su2", "gfx1151"))
        .unwrap();
    fs::write(c.chemin_code("su2", "gfx1151", &k, "hsaco"), &code[..512]).unwrap();
    assert!(c.lire("su2", "gfx1151", &k, "hsaco").is_none());
    assert!(fichiers(&d).is_empty());
}

#[test]
fn json_absent_illisible_ou_incoherent_evince() {
    for variante in ["absent", "illisible", "autre_cle"] {
        let d = repertoire_de_test(&format!("json-{variante}"));
        let c = Cache::nouveau(&d, 1 << 20);
        let (k, code) = (cle_n(5), code_n(5, 256));
        c.ecrire("su2", "gfx1151", &k, "hsaco", &code, meta("su2", "gfx1151"))
            .unwrap();
        let pj = c.chemin_meta("su2", "gfx1151", &k);
        match variante {
            "absent" => fs::remove_file(&pj).unwrap(),
            "illisible" => fs::write(&pj, b"{ pas du json").unwrap(),
            _ => {
                let mut m: MetaEntree = serde_json::from_slice(&fs::read(&pj).unwrap()).unwrap();
                m.cle = "0".repeat(64);
                fs::write(&pj, serde_json::to_vec(&m).unwrap()).unwrap();
            }
        }
        assert!(c.lire("su2", "gfx1151", &k, "hsaco").is_none(), "variante {variante}");
        assert!(fichiers(&d).is_empty(), "variante {variante} : {:?}", fichiers(&d));
    }
}

#[test]
fn ecriture_refuse_un_code_vide() {
    let d = repertoire_de_test("vide");
    let c = Cache::nouveau(&d, 1 << 20);
    assert!(c
        .ecrire("su2", "gfx1151", &cle_n(6), "hsaco", &[], meta("su2", "gfx1151"))
        .is_err());
    assert!(
        fichiers(&d).iter().all(|f| !f.ends_with(".hsaco")),
        "aucun code objet écrit"
    );
}

#[test]
fn plafond_evince_les_plus_anciens_en_lru() {
    let d = repertoire_de_test("lru");
    // Entrées de 100 000 octets (+ json) ; plafond ≈ 4,4 entrées : quatre tiennent, la cinquième déborde.
    let taille = 100_000usize;
    let c = Cache::nouveau(&d, 440_000);
    let cles: Vec<String> = (10..14).map(cle_n).collect();
    for (i, k) in cles.iter().enumerate() {
        c.ecrire("su2", "gfx1151", k, "hsaco", &code_n(i, taille), meta("su2", "gfx1151"))
            .unwrap();
        // Ancienneté décroissante : l'entrée 0 est la plus vieille.
        dater(&c.chemin_code("su2", "gfx1151", k, "hsaco"), 1000 - 100 * i as u64);
    }
    assert!(
        c.taille_totale() <= 440_000,
        "déjà sous le plafond avant la cinquième écriture"
    );
    // Une lecture rafraîchit l'entrée 0 : elle devient la plus récente.
    assert!(c.lire("su2", "gfx1151", &cles[0], "hsaco").is_some());
    // Cinquième écriture : dépassement ⇒ l'entrée 1 (la plus ancienne non rafraîchie) est évincée.
    let k5 = cle_n(14);
    c.ecrire(
        "su2",
        "gfx1151",
        &k5,
        "hsaco",
        &code_n(14, taille),
        meta("su2", "gfx1151"),
    )
    .unwrap();
    assert!(c.taille_totale() <= 440_000, "plafond respecté : {}", c.taille_totale());
    assert!(
        c.lire("su2", "gfx1151", &cles[1], "hsaco").is_none(),
        "entrée 1 évincée (la plus ancienne)"
    );
    assert!(
        c.lire("su2", "gfx1151", &cles[0], "hsaco").is_some(),
        "entrée 0 rafraîchie : conservée"
    );
    assert!(
        c.lire("su2", "gfx1151", &k5, "hsaco").is_some(),
        "entrée neuve conservée"
    );
}

#[test]
fn plafond_applique_aussi_a_la_demande() {
    let d = repertoire_de_test("plafond-direct");
    let grand = Cache::nouveau(&d, 1 << 30);
    for n in 20..25 {
        grand
            .ecrire(
                "x",
                "gfx1151",
                &cle_n(n),
                "hsaco",
                &code_n(n, 50_000),
                meta("x", "gfx1151"),
            )
            .unwrap();
    }
    let avant = grand.taille_totale();
    assert!(avant > 250_000);
    let petit = Cache::nouveau(&d, 120_000);
    let evinces = petit.appliquer_plafond();
    assert!(evinces >= 3, "évincées : {evinces}");
    assert!(petit.taille_totale() <= 120_000);
    assert_eq!(petit.appliquer_plafond(), 0, "idempotent sous le plafond");
}

#[test]
fn plafond_zero_ne_garde_rien() {
    let d = repertoire_de_test("zero");
    let c = Cache::nouveau(&d, 0);
    c.ecrire(
        "x",
        "gfx1151",
        &cle_n(30),
        "hsaco",
        &code_n(30, 100),
        meta("x", "gfx1151"),
    )
    .unwrap();
    assert!(fichiers(&d).is_empty());
}

#[test]
fn cle_depend_de_chaque_ingredient() {
    let args = vec!["-O3".to_string(), "-std=c++17".to_string()];
    let base = cache::cle("texte", &args, "gfx1151", "hiprtc=9.0");
    assert_eq!(base.len(), 64);
    assert_eq!(
        base,
        cache::cle("texte", &args, "gfx1151", "hiprtc=9.0"),
        "déterministe"
    );
    let autres = [
        cache::cle("texte2", &args, "gfx1151", "hiprtc=9.0"),
        cache::cle("texte", &args[..1], "gfx1151", "hiprtc=9.0"),
        cache::cle("texte", &[args[1].clone(), args[0].clone()], "gfx1151", "hiprtc=9.0"),
        cache::cle("texte", &args, "gfx1100", "hiprtc=9.0"),
        cache::cle("texte", &args, "gfx1151", "hiprtc=9.1"),
    ];
    for (i, a) in autres.iter().enumerate() {
        assert_ne!(a, &base, "ingrédient {i}");
    }
}

#[test]
fn cle_valeur_d_or() {
    // Flux d'octets documenté dans cache.rs : champs précédés de leur longueur (u64 LE) — version du format,
    // texte, nombre d'arguments, chaque argument, arch, version du compilateur. Recalculé ici à la main.
    let mut flux: Vec<u8> = Vec::new();
    let mut champ = |o: &[u8]| {
        flux.extend_from_slice(&(o.len() as u64).to_le_bytes());
        flux.extend_from_slice(o);
    };
    champ(&cache::VERSION_FORMAT.to_le_bytes());
    champ(b"abc");
    champ(&1u64.to_le_bytes());
    champ(b"-O3");
    champ(b"sm_120");
    champ(b"nvrtc=13.3");
    let attendu = kvu::heritage::manifest::sha256_hex(&flux);
    assert_eq!(cache::cle("abc", &["-O3".to_string()], "sm_120", "nvrtc=13.3"), attendu);
}

#[test]
fn ecriture_atomique_sans_temporaire_restant() {
    let d = repertoire_de_test("atomique");
    let c = Cache::nouveau(&d, 1 << 20);
    for n in 40..46 {
        c.ecrire("x", "sm_120", &cle_n(n), "cubin", &code_n(n, 3000), meta("x", "sm_120"))
            .unwrap();
    }
    assert!(
        fichiers(&d).iter().all(|f| !f.starts_with(".tmp-")),
        "{:?}",
        fichiers(&d)
    );
    assert_eq!(fichiers(&d).len(), 12);
}

#[test]
fn ecritures_concurrentes_de_la_meme_entree() {
    let d = repertoire_de_test("concurrent");
    let k = cle_n(50);
    let code = code_n(50, 20_000);
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| {
                let c = Cache::nouveau(&d, 1 << 20);
                for _ in 0..10 {
                    c.ecrire("x", "sm_120", &k, "cubin", &code, meta("x", "sm_120"))
                        .unwrap();
                    if let Some(e) = c.lire("x", "sm_120", &k, "cubin") {
                        assert_eq!(e.code, code, "jamais d'entrée à moitié écrite");
                    }
                }
            });
        }
    });
    let c = Cache::nouveau(&d, 1 << 20);
    assert_eq!(c.lire("x", "sm_120", &k, "cubin").unwrap().code, code);
}

#[test]
fn temporaires_abandonnes_nettoyes() {
    let d = repertoire_de_test("tmp");
    let c = Cache::nouveau(&d, 1 << 20);
    let vieux = d.join(".tmp-1-1-x.json");
    let recent = d.join(".tmp-2-2-y.json");
    fs::write(&vieux, b"x").unwrap();
    fs::write(&recent, b"y").unwrap();
    dater(&vieux, 7200);
    c.appliquer_plafond();
    assert!(!vieux.exists(), "temporaire de plus d'une heure supprimé");
    assert!(
        recent.exists(),
        "temporaire récent conservé (écriture peut-être en cours)"
    );
}

#[test]
fn cache_depuis_env_respecte_la_desactivation() {
    // Variables d'environnement propres à ce test : un seul test les modifie dans ce binaire.
    std::env::set_var("KEVINOTRON_CACHE_DIR", "/chemin/inexistant/kvu");
    std::env::set_var("KEVINOTRON_CACHE_MAX_MO", "7");
    std::env::remove_var("KEVINOTRON_CACHE_DISABLE");
    let c = Cache::depuis_env().expect("actif");
    assert_eq!(c.dir(), Path::new("/chemin/inexistant/kvu"));
    assert_eq!(c.max_octets(), 7 << 20);
    std::env::set_var("KEVINOTRON_CACHE_DISABLE", "1");
    assert!(Cache::depuis_env().is_none());
    std::env::set_var("KEVINOTRON_CACHE_DISABLE", "0");
    assert!(Cache::depuis_env().is_some());
    std::env::remove_var("KEVINOTRON_CACHE_DISABLE");
    std::env::remove_var("KEVINOTRON_CACHE_DIR");
    std::env::remove_var("KEVINOTRON_CACHE_MAX_MO");
}
