//! Portes NPY du socle geom : ordre C, compatibilité historique et refus des métadonnées incohérentes.

#[path = "donnees/stockage/support.rs"]
mod support;

use kvu::commun::{ChampHote, Erreur};
use kvu::geom::{
    ecrire_geometrie, examiner_voisinage, examiner_voisinage_avec, geometrie_voisine,
    lire_geometrie, ChampGeo, Chemin, GeoSpec, MutantVoisinage, SourceBord, Torsion,
    NOM_FICHIER_GEOMETRIE,
};
use kvu::heritage::manifest::sha256_fichier;
use kvu::stockage::garde::GardeSsd;
use kvu::stockage::npy::{self, EnteteNpy, FormatChamp};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use support::Scratch;

fn comparer<T: PartialEq>(obtenu: T, attendu: T, comparaisons: &mut usize) -> usize {
    *comparaisons += 1;
    usize::from(obtenu != attendu)
}

fn terminer(nom: &str, ecarts: usize, comparaisons: usize) {
    assert!(comparaisons > 0, "{nom} : aucune comparaison");
    assert_eq!(ecarts, 0, "{nom}");
    println!("PASS {nom} {comparaisons}");
}

fn ecarts_refus<T>(resultat: Result<T, Erreur>, comparaisons: &mut usize) -> usize {
    comparer(
        resultat.err().map(|e| e.code_sortie()),
        Some(2),
        comparaisons,
    )
}

fn dossier(sc: &Scratch, nom: &str) -> PathBuf {
    let chemin = sc.p(nom);
    fs::create_dir_all(&chemin).unwrap();
    chemin
}

fn champ(cotes: [usize; 4]) -> ChampGeo {
    let mut c = ChampGeo::froid(GeoSpec::anisotrope(cotes, 2)).unwrap();
    for (li, q) in c.q.chunks_exact_mut(4).enumerate() {
        let theta = 1e-3 * (li + 1) as f64;
        q.copy_from_slice(&[theta.cos(), 0.0, 0.0, theta.sin()]);
    }
    c
}

/// Vérifie les octets bruts q64 par énumération (μ, x0, x1, x2, x3), sans utiliser Tables.
fn ecarts_q64(brut: &[u8], e: &EnteteNpy, c: &[usize; 4], comparaisons: &mut usize) -> usize {
    let mut ecarts = 0;
    let mut compteur = 0;
    for mu in 0..4 {
        for x0 in 0..c[0] {
            for x1 in 0..c[1] {
                for x2 in 0..c[2] {
                    for x3 in 0..c[3] {
                        let li = (((mu * c[0] + x0) * c[1] + x1) * c[2] + x2) * c[3] + x3;
                        ecarts += comparer(li, compteur, comparaisons);
                        let theta = 1e-3 * (li + 1) as f64;
                        let attendu = [theta.cos(), 0.0, 0.0, theta.sin()];
                        for (k, valeur) in attendu.iter().enumerate() {
                            let position = e.octets_entete + 32 * li + 8 * k;
                            let octets = valeur.to_le_bytes();
                            ecarts += comparer(
                                &brut[position..position + 8],
                                octets.as_slice(),
                                comparaisons,
                            );
                        }
                        compteur += 1;
                    }
                }
            }
        }
    }
    ecarts += comparer(compteur, 4 * c.iter().product::<usize>(), comparaisons);
    ecarts
}

#[test]
fn g_npy_anisotrope() {
    let sc = Scratch::nouveau("geom_npy");
    let garde = GardeSsd::nouvelle(0.0);
    let mut n = 0;
    let mut ecarts = 0;
    for cotes in [[2, 3, 4, 5], [4, 4, 2, 2]] {
        let c = champ(cotes);
        for format in [FormatChamp::C16, FormatChamp::Q64] {
            let d = dossier(&sc, &format!("{cotes:?}_{}", format.nom()));
            let chemin = d.join("champ.npy");
            let resume = npy::ecrire_champ_geo(&chemin, &c, format, &garde).unwrap();
            ecarts += comparer(d.join(NOM_FICHIER_GEOMETRIE).exists(), false, &mut n);
            let geometrie = ecrire_geometrie(&d, &c.geo).unwrap();
            ecarts += comparer(geometrie, d.join(NOM_FICHIER_GEOMETRIE), &mut n);
            ecarts += comparer(d.join("geometrie.json.part").exists(), false, &mut n);
            ecarts += comparer(d.join("champ.npy.part").exists(), false, &mut n);
            let lu = npy::lire_champ_geo(&chemin, 0).unwrap();
            ecarts += comparer(&lu, &c, &mut n);
            ecarts += comparer(&lu.geo, &c.geo, &mut n);
            ecarts += comparer(lu.q.len(), c.q.len(), &mut n);
            for (&a, &b) in lu.q.iter().zip(&c.q) {
                ecarts += comparer(a.to_bits(), b.to_bits(), &mut n);
            }
            let brut = fs::read(&chemin).unwrap();
            let e = npy::lire_entete(&mut &brut[..]).unwrap();
            let [c0, c1, c2, c3] = cotes;
            let forme = match format {
                FormatChamp::C16 => vec![4, c0, c1, c2, c3, 2, 2],
                FormatChamp::Q64 => vec![4, c0, c1, c2, c3, 4],
            };
            ecarts += comparer(&e.forme, &forme, &mut n);
            ecarts += comparer(e.fortran, false, &mut n);
            ecarts += comparer(
                npy::disposition_champ_geo(&e).unwrap(),
                (format, cotes),
                &mut n,
            );
            ecarts += comparer(resume.octets, brut.len() as u64, &mut n);
            ecarts += comparer(
                resume.sha256,
                Some(sha256_fichier(&chemin).unwrap()),
                &mut n,
            );
            if format == FormatChamp::Q64 {
                ecarts += ecarts_q64(&brut, &e, &cotes, &mut n);
            }
        }
    }
    terminer("G-NPY-ANISOTROPE", ecarts, n);
}

#[test]
fn g_npy_c0_identique() {
    let sc = Scratch::nouveau("geom_npy_c0");
    let garde = GardeSsd::nouvelle(0.0);
    let h = ChampHote {
        l: 4,
        q: champ([4; 4]).q,
    };
    let c = ChampGeo::depuis_hote(h.clone(), 3).unwrap();
    let mut n = 0;
    let mut ecarts = comparer(c.geo.chemin(), Chemin::C0, &mut n);
    ecarts += comparer(ChampHote::try_from(c.clone()).unwrap(), h.clone(), &mut n);
    for format in [FormatChamp::C16, FormatChamp::Q64] {
        let historique = sc.p(&format!("historique_{}.npy", format.nom()));
        let geometrie = sc.p(&format!("geometrie_{}.npy", format.nom()));
        npy::ecrire_champ(&historique, &h, format, &garde).unwrap();
        npy::ecrire_champ_geo(&geometrie, &c, format, &garde).unwrap();
        ecarts += comparer(
            fs::read(&geometrie).unwrap(),
            fs::read(&historique).unwrap(),
            &mut n,
        );
        ecarts += comparer(
            npy::entete_champ_cotes(&[4; 4], format),
            npy::entete_champ(4, format),
            &mut n,
        );
        let lu = npy::lire_champ_geo(&historique, 3).unwrap();
        ecarts += comparer(lu.geo.chemin(), Chemin::C0, &mut n);
        ecarts += comparer(lu.geo.dir_temps, 3, &mut n);
        ecarts += comparer(lu.q, npy::lire_champ(&historique).unwrap().q, &mut n);
    }
    terminer("G-NPY-C0-OCTETS", ecarts, n);
}

#[test]
fn g_refus_fichiers() {
    let sc = Scratch::nouveau("geom_refus");
    let garde = GardeSsd::nouvelle(0.0);
    let mut n = 0;
    let mut ecarts = 0;

    let d = dossier(&sc, "anisotrope_sans_geometrie");
    let c = champ([2, 3, 4, 5]);
    let chemin = d.join("champ.npy");
    npy::ecrire_champ_geo(&chemin, &c, FormatChamp::Q64, &garde).unwrap();
    ecarts += ecarts_refus(npy::lire_champ_geo(&chemin, 3), &mut n);
    ecarts += ecarts_refus(ChampHote::try_from(c), &mut n);

    let d = dossier(&sc, "torsion_sans_geometrie");
    let c = champ([4; 4]);
    let chemin = d.join("champ.npy");
    npy::ecrire_champ_geo(&chemin, &c, FormatChamp::C16, &garde).unwrap();
    fs::write(d.join("torsion.json"), br#"{"plans": ["01"]}"#).unwrap();
    ecarts += ecarts_refus(npy::lire_champ_geo(&chemin, 3), &mut n);

    let d = dossier(&sc, "cotes_incompatibles");
    let chemin = d.join("champ.npy");
    npy::ecrire_champ_geo(&chemin, &c, FormatChamp::Q64, &garde).unwrap();
    ecrire_geometrie(&d, &GeoSpec::anisotrope([4, 4, 4, 6], 3)).unwrap();
    ecarts += ecarts_refus(npy::lire_champ_geo(&chemin, 3), &mut n);

    let d = dossier(&sc, "torsions_contradictoires");
    let chemin = d.join("champ.npy");
    let mut c = c;
    c.geo.torsion = Torsion::analyser("01").unwrap();
    npy::ecrire_champ_geo(&chemin, &c, FormatChamp::Q64, &garde).unwrap();
    ecrire_geometrie(&d, &c.geo).unwrap();
    fs::write(d.join("torsion.json"), br#"{"plans": ["23"]}"#).unwrap();
    ecarts += ecarts_refus(npy::lire_champ_geo(&chemin, 3), &mut n);
    terminer("G-REFUS-FICHIERS", ecarts, n);
}

#[test]
fn mutant_m6_cotes_geometrie_permutes() {
    let sc = Scratch::nouveau("geom_mutant_m6");
    let c = champ([2, 3, 4, 5]);
    let chemin = sc.p("champ.npy");
    npy::ecrire_champ_geo(&chemin, &c, FormatChamp::Q64, &GardeSsd::nouvelle(0.0)).unwrap();
    let mut spec = c.geo;
    spec.cotes.swap(1, 2);
    ecrire_geometrie(&sc.0, &spec).unwrap();
    let erreur = npy::lire_champ_geo(&chemin, 3).unwrap_err();
    let mut n = 0;
    assert_eq!(comparer(erreur.code_sortie(), 2, &mut n), 0);
    let ecarts = comparer(erreur.code_sortie(), 0, &mut n);
    assert!(ecarts > 0, "M6 : permutation des côtés non détectée");
    assert!(n > 0);
    println!(
        "PASS M6-COTES-GEOMETRIE-PERMUTES {n} (mutant ROUGE : refus code {})",
        erreur.code_sortie()
    );
}

/// Configuration vide (seule son existence compte pour la recherche des métadonnées voisines).
fn cfg_vide(d: &Path) -> PathBuf {
    let cfg = d.join("champ.npy");
    fs::write(&cfg, b"").unwrap();
    cfg
}

#[test]
fn g_geometrie_voisine_parent_et_accord() {
    let sc = Scratch::nouveau("geom_voisine");
    let parent = dossier(&sc, "experience");
    let chaine = parent.join("c0");
    fs::create_dir_all(&chaine).unwrap();
    let cfg = cfg_vide(&chaine);
    let mut n = 0;
    let mut ecarts = comparer(geometrie_voisine(&cfg).unwrap(), None, &mut n);
    // géométrie du lot dans le parent : trouvée depuis c0/
    let tordue = GeoSpec::cubique(4, 3).avec_torsion(Torsion::analyser("01").unwrap());
    let chemin_parent = ecrire_geometrie(&parent, &tordue).unwrap();
    let parent_canon = fs::canonicalize(&chemin_parent).unwrap();
    ecarts += comparer(
        geometrie_voisine(&cfg).unwrap(),
        Some((tordue.clone(), parent_canon.clone())),
        &mut n,
    );
    ecarts += comparer(
        lire_geometrie(&chemin_parent).unwrap(),
        tordue.clone(),
        &mut n,
    );
    // même géométrie aussi dans c0 : accord, le fichier le plus proche est cité
    let chemin_local = fs::canonicalize(ecrire_geometrie(&chaine, &tordue).unwrap()).unwrap();
    ecarts += comparer(
        geometrie_voisine(&cfg).unwrap(),
        Some((tordue.clone(), chemin_local.clone())),
        &mut n,
    );
    // torsion historique concordante (forme M4 complète) : accord
    let m4 = json!({"plans": ["01"], "pile": "coins (L-1, L-1) de chaque plan tordu (convention kvu M4)", "facteur": -1});
    fs::write(parent.join("torsion.json"), m4.to_string()).unwrap();
    ecarts += comparer(
        geometrie_voisine(&cfg).unwrap(),
        Some((tordue.clone(), chemin_local)),
        &mut n,
    );
    // torsion historique discordante : refus
    fs::write(parent.join("torsion.json"), br#"{"plans": ["23"]}"#).unwrap();
    ecarts += ecarts_refus(geometrie_voisine(&cfg), &mut n);
    fs::remove_file(parent.join("torsion.json")).unwrap();
    // deux geometrie.json différents (parent et c0) : refus, plus de « le plus proche gagne »
    ecrire_geometrie(&parent, &GeoSpec::cubique(4, 3)).unwrap();
    ecarts += ecarts_refus(geometrie_voisine(&cfg), &mut n);
    terminer("G-GEOMETRIE-VOISINE", ecarts, n);
}

/// O1 (relecture du 11/11) : le bloc canonique d'un manifest.json est lu, et toutes les déclarations sont confrontées,
/// même quand un geometrie.json existe.
#[test]
fn g_o1_declarations_confrontees() {
    let sc = Scratch::nouveau("geom_o1");
    let garde = GardeSsd::nouvelle(0.0);
    let tordue = GeoSpec::cubique(4, 3).avec_torsion(Torsion::analyser("01").unwrap());
    let mut n = 0;
    let mut ecarts = 0;

    // (a) NPY cubique pair + manifest.json au bloc canonique C1 tordu, sans geometrie.json : JAMAIS C0
    let d = dossier(&sc, "a_bloc_manifeste");
    let c = ChampGeo::depuis_hote(
        ChampHote {
            l: 4,
            q: champ([4; 4]).q,
        },
        3,
    )
    .unwrap();
    let cfg = d.join("champ.npy");
    npy::ecrire_champ_geo(&cfg, &c, FormatChamp::C16, &garde).unwrap();
    fs::write(
        d.join("manifest.json"),
        json!({"mode": "su2-gen-cotes", "geometrie": tordue.bloc()}).to_string(),
    )
    .unwrap();
    let lu = npy::lire_champ_geo(&cfg, 3).unwrap();
    ecarts += comparer(lu.geo.chemin(), Chemin::C1, &mut n);
    ecarts += comparer(&lu.geo, &tordue, &mut n);
    ecarts += ecarts_refus(ChampHote::try_from(lu), &mut n);
    let source = examiner_voisinage(&cfg).unwrap().spec.map(|(_, p)| p);
    ecarts += comparer(
        source,
        Some(fs::canonicalize(d.join("manifest.json")).unwrap()),
        &mut n,
    );

    // (b)-(e) geometrie.json C0 contredit par une déclaration historique ou canonique : refus code 2
    for (i, (nom, contenu)) in [
        ("manifest.json", json!({"torsion": "01"})),
        (
            "manifest.json",
            json!({"mode": "su2-bord", "bord": [{"chaine": 0}]}),
        ),
        ("config_run.json", json!({"masque": [1]})),
        ("manifest.json", json!({"geometrie": tordue.bloc()})),
        ("torsion.json", json!({"plans": ["01"]})),
        ("manifest.json", json!({"mode": "su2-boite"})),
    ]
    .iter()
    .enumerate()
    {
        let d = dossier(&sc, &format!("contredit_{i}"));
        let cfg = cfg_vide(&d);
        ecrire_geometrie(&d, &GeoSpec::cubique(4, 3)).unwrap();
        fs::write(d.join(nom), contenu.to_string()).unwrap();
        ecarts += ecarts_refus(geometrie_voisine(&cfg), &mut n);
    }

    // (e2) R2-3 historique (hyperplan x0 = 0) contre une géométrie dont le bord fixe est sur un AUTRE axe : refus code 2
    let d = dossier(&sc, "e2_r23_autre_axe");
    let cfg = cfg_vide(&d);
    let mut axe1 = GeoSpec::cubique(4, 3);
    axe1.bords[1] = kvu::geom::Bord::Fixe {
        valeurs: SourceBord::Froid,
    };
    ecrire_geometrie(&d, &axe1).unwrap();
    fs::write(
        d.join("manifest.json"),
        json!({"mode": "su2-bord", "bord": [{"chaine": 0, "bc": "cold"}]}).to_string(),
    )
    .unwrap();
    ecarts += ecarts_refus(geometrie_voisine(&cfg), &mut n);

    // (f) déclarations concordantes : acceptées
    let d = dossier(&sc, "f_hyperplan_r23");
    let cfg = cfg_vide(&d);
    let r23 = GeoSpec::hyperplan(4, 3, SourceBord::Chaud { graine: 3 });
    ecrire_geometrie(&d, &r23).unwrap();
    fs::write(
        d.join("manifest.json"),
        json!({"mode": "su2-bord", "bord": [{"chaine": 0}], "geometrie": "texte libre R2-3"})
            .to_string(),
    )
    .unwrap();
    ecarts += comparer(
        geometrie_voisine(&cfg).unwrap().map(|(g, _)| g),
        Some(r23),
        &mut n,
    );
    let d = dossier(&sc, "f_tordue");
    let cfg = cfg_vide(&d);
    ecrire_geometrie(&d, &tordue).unwrap();
    fs::write(
        d.join("manifest.json"),
        json!({"torsion": {"plans": ["01"], "facteur": -1}, "geometrie": tordue.bloc()})
            .to_string(),
    )
    .unwrap();
    ecarts += comparer(
        geometrie_voisine(&cfg).unwrap().map(|(g, _)| g),
        Some(tordue.clone()),
        &mut n,
    );

    // (g) bloc « geometrie » historique SANS schema (rg : nom de pavage) : non interprété, configuration C0 admise
    let d = dossier(&sc, "g_rg");
    let cfg = cfg_vide(&d);
    fs::write(
        d.join("manifest.json"),
        json!({"geometrie": "3centre"}).to_string(),
    )
    .unwrap();
    ecarts += comparer(geometrie_voisine(&cfg).unwrap(), None, &mut n);

    // (h) bloc canonique mal formé : refus code 2
    let d = dossier(&sc, "h_bloc_mal_forme");
    let cfg = cfg_vide(&d);
    fs::write(
        d.join("manifest.json"),
        json!({"geometrie": {"schema": "kvu-geometrie/1", "cotes": [4, 4, 4, 4]}}).to_string(),
    )
    .unwrap();
    ecarts += ecarts_refus(geometrie_voisine(&cfg), &mut n);
    terminer("G-O1-DECLARATIONS-CONFRONTEES", ecarts, n);
}

/// Mutant O1 : bloc canonique du manifeste ignoré ⇒ la configuration (a) passerait pour C0 : la porte doit le voir.
#[test]
fn mutant_m7_bloc_manifeste_ignore() {
    let sc = Scratch::nouveau("geom_m7");
    let cfg = cfg_vide(&sc.0);
    let tordue = GeoSpec::cubique(4, 3).avec_torsion(Torsion::analyser("01").unwrap());
    fs::write(
        sc.p("manifest.json"),
        json!({"geometrie": tordue.bloc()}).to_string(),
    )
    .unwrap();
    let mut n = 0;
    let bon = examiner_voisinage_avec(&cfg, MutantVoisinage::Aucun)
        .unwrap()
        .spec
        .map(|(g, _)| g);
    assert_eq!(comparer(bon, Some(tordue.clone()), &mut n), 0);
    let mutant = examiner_voisinage_avec(&cfg, MutantVoisinage::IgnoreBlocManifeste)
        .unwrap()
        .spec
        .map(|(g, _)| g);
    let ecarts = comparer(mutant, Some(tordue), &mut n);
    assert!(ecarts > 0, "M7 : bloc de manifeste ignoré non détecté");
    println!("PASS M7-BLOC-MANIFESTE-IGNORE {n} (mutant ROUGE : {ecarts} écarts)");
}

/// O4 : fichier absent, déclaration triviale VALIDE et déclaration MAL FORMÉE sont distingués.
#[test]
fn g_refus_metadonnees_historiques() {
    let sc = Scratch::nouveau("geom_meta");
    let mut n = 0;
    let mut ecarts = 0;
    for (i, v) in [
        json!({"torsion": {"plans": ["01"]}}),
        json!({"torsion": "01"}),
        json!({"config_run": {"torsion": ["01"]}}),
        json!({"bord": "cold"}),
        json!({"masque": [1]}),
        json!({"config_run": {"bord": {}}}),
        json!({"config_run": {"masque": true}}),
        json!({"mode": "su2-bord"}),
        json!({"mode": "su2-boite"}),
        json!({"torsion": {"plan": ["01"]}}),
        json!({"torsion": {}}),
        json!({"torsion": 3}),
        json!([1, 2]),
    ]
    .iter()
    .enumerate()
    {
        for nom in ["manifest.json", "config_run.json"] {
            let d = dossier(&sc, &format!("refus_{i}_{nom}"));
            fs::write(d.join(nom), v.to_string()).unwrap();
            ecarts += ecarts_refus(geometrie_voisine(&cfg_vide(&d)), &mut n);
        }
    }
    for v in [Value::Null, json!(false), json!([]), json!("")] {
        let d = dossier(&sc, &format!("neutre_{n}"));
        let declaration = json!({"bord": v, "masque": v, "torsion": Value::Null,
                                 "config_run": {"bord": v, "masque": v, "torsion": Value::Null}});
        fs::write(d.join("manifest.json"), declaration.to_string()).unwrap();
        ecarts += comparer(geometrie_voisine(&cfg_vide(&d)).unwrap(), None, &mut n);
    }
    // torsion.json : triviale valide (acceptée), mal formée (refusée, code 2)
    for v in [
        json!("aucune"),
        json!({"plans": []}),
        json!({"plans": [], "pile": "coins (L-1, L-1) de chaque plan tordu (convention kvu M4)", "facteur": -1}),
        json!({"plans": [false, false, false, false, false, false]}),
        json!([]),
    ] {
        let d = dossier(&sc, &format!("torsion_triviale_{n}"));
        fs::write(d.join("torsion.json"), v.to_string()).unwrap();
        ecarts += comparer(geometrie_voisine(&cfg_vide(&d)).unwrap(), None, &mut n);
    }
    for v in [
        Value::Null,
        json!(false),
        json!({}),
        json!({"plan": ["01"]}),
        json!({"plans": 1}),
        json!({"plans": ["0x"]}),
        json!({"plans": ["01"], "inconnu": 1}),
        json!({"plans": [], "facteur": 1}),
        json!({"plans": [], "pile": 2}),
        json!({"plans": [true, false]}),
        json!(""),
    ] {
        let d = dossier(&sc, &format!("torsion_mal_formee_{n}"));
        fs::write(d.join("torsion.json"), v.to_string()).unwrap();
        ecarts += ecarts_refus(geometrie_voisine(&cfg_vide(&d)), &mut n);
    }
    // fichier de configuration absent : erreur d'entrée-sortie (code 4), jamais « C0 supposé »
    let absent = geometrie_voisine(&sc.p("absent/champ.npy"))
        .err()
        .map(|e| e.code_sortie());
    ecarts += comparer(absent, Some(4), &mut n);
    terminer("G-REFUS-METADONNEES", ecarts, n);
}

#[test]
fn g_refus_format_et_su2() {
    let sc = Scratch::nouveau("geom_npy_format");
    let garde = GardeSsd::nouvelle(0.0);
    let c = champ([2, 3, 4, 5]);
    ecrire_geometrie(&sc.0, &c.geo).unwrap();
    let chemin = sc.p("champ.npy");
    npy::ecrire_champ_geo(&chemin, &c, FormatChamp::Q64, &garde).unwrap();
    let brut = fs::read(&chemin).unwrap();
    let e = npy::lire_entete(&mut &brut[..]).unwrap();
    let mut n = 0;
    let mut ecarts = 0;
    fs::write(sc.p("tronque.npy"), &brut[..brut.len() - 1]).unwrap();
    ecarts += comparer(
        matches!(
            npy::lire_champ_geo(&sc.p("tronque.npy"), 3),
            Err(Erreur::Format(_))
        ),
        true,
        &mut n,
    );
    let mut trop = brut.clone();
    trop.push(0);
    fs::write(sc.p("trop.npy"), trop).unwrap();
    ecarts += comparer(
        matches!(
            npy::lire_champ_geo(&sc.p("trop.npy"), 3),
            Err(Erreur::Format(_))
        ),
        true,
        &mut n,
    );
    let mut mauvais = brut;
    mauvais[e.octets_entete..e.octets_entete + 8].copy_from_slice(&f64::NAN.to_le_bytes());
    fs::write(sc.p("nan.npy"), mauvais).unwrap();
    ecarts += comparer(
        matches!(
            npy::lire_champ_geo(&sc.p("nan.npy"), 3),
            Err(Erreur::Format(_))
        ),
        true,
        &mut n,
    );
    let mut non_unitaire = c;
    non_unitaire.q[..4].fill(0.0);
    let ecriture = npy::ecrire_champ_geo(&sc.p("nul.npy"), &non_unitaire, FormatChamp::Q64, &garde);
    ecarts += comparer(matches!(ecriture, Err(Erreur::Validation(_))), true, &mut n);
    ecarts += comparer(sc.p("nul.npy").exists(), false, &mut n);
    ecarts += comparer(sc.p("nul.npy.part").exists(), false, &mut n);
    terminer("G-REFUS-FORMAT-SU2", ecarts, n);
}

/// O8 : geometrie.json syntaxiquement invalide ⇒ code 2 ; erreur d'entrée-sortie réelle ⇒ code 4 ; le chemin est cité.
#[test]
fn g_refus_json_geometrie() {
    let sc = Scratch::nouveau("geom_json_refus");
    let chemin = sc.p(NOM_FICHIER_GEOMETRIE);
    let mut n = 0;
    let mut ecarts = 0;
    fs::write(&chemin, "{").unwrap();
    let erreur = lire_geometrie(&chemin).unwrap_err();
    ecarts += comparer(erreur.code_sortie(), 2, &mut n);
    ecarts += comparer(
        erreur.to_string().contains(&chemin.display().to_string()),
        true,
        &mut n,
    );
    ecarts += ecarts_refus(geometrie_voisine(&cfg_vide(&sc.0)), &mut n);
    let mut v = GeoSpec::cubique(4, 3).bloc();
    v.as_object_mut().unwrap().remove("dir_temps");
    fs::write(&chemin, v.to_string()).unwrap();
    ecarts += ecarts_refus(lire_geometrie(&chemin), &mut n);
    // un dossier nommé geometrie.json : lecture impossible, erreur d'E/S (code 4)
    let sc2 = Scratch::nouveau("geom_json_es");
    fs::create_dir_all(sc2.p(NOM_FICHIER_GEOMETRIE)).unwrap();
    ecarts += comparer(
        lire_geometrie(&sc2.p(NOM_FICHIER_GEOMETRIE))
            .err()
            .map(|e| e.code_sortie()),
        Some(4),
        &mut n,
    );
    ecarts += comparer(
        geometrie_voisine(&cfg_vide(&sc2.0))
            .err()
            .map(|e| e.code_sortie()),
        Some(4),
        &mut n,
    );
    terminer("G-REFUS-JSON-FICHIER", ecarts, n);
}
