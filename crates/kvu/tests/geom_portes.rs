//! Portes CPU du socle geom et mutants vérifiant la sensibilité des comparaisons.

use kvu::commun::Erreur;
use kvu::geom::{
    schema, Bord, Chemin, Coloriage, Couture, GeoSpec, RegionSpec, SourceBord, SourceMasque,
    Tables, Torsion, PLANS, SCHEMA_TEXTE,
};
use kvu::heritage::su2_hb::Geo;
use kvu::su2::boite::{masque_boite, masque_gel_x0_spatial, GeometrieBoite};
use serde_json::{json, Value};

fn comparer<T: PartialEq>(obtenu: T, attendu: T, comparaisons: &mut usize) -> usize {
    *comparaisons += 1;
    usize::from(obtenu != attendu)
}

fn terminer(nom: &str, ecarts: usize, comparaisons: usize) {
    assert!(comparaisons > 0, "{nom} : aucune comparaison");
    assert_eq!(ecarts, 0, "{nom}");
    println!("PASS {nom} {comparaisons}");
}

fn terminer_mutant(nom: &str, ecarts: usize, comparaisons: usize) {
    assert!(comparaisons > 0, "{nom} : aucune comparaison");
    assert!(ecarts > 0, "{nom} : mutant non détecté");
    println!("PASS {nom} {comparaisons} (mutant ROUGE : {ecarts} écarts)");
}

fn ecarts_c0(t: &Tables, l: usize, comparaisons: &mut usize) -> usize {
    let g = Geo::new(l);
    let mut ecarts = comparer(t.cotes, [l; 4], comparaisons);
    ecarts += comparer(t.pas, g.st, comparaisons);
    ecarts += comparer(t.v, g.v, comparaisons);
    ecarts += comparer(t.coef.is_none(), true, comparaisons);
    ecarts += comparer(t.masque.is_none(), true, comparaisons);
    ecarts += comparer(t.explicite.is_none(), true, comparaisons);
    ecarts += comparer(t.coloriage, Coloriage::Parite, comparaisons);
    ecarts += comparer(t.classes.len(), 2, comparaisons);
    let voisins = t.voisins();
    ecarts += comparer(voisins.len(), g.v, comparaisons);
    for (s, ligne) in voisins.iter().enumerate() {
        let x = g.coords(s);
        ecarts += comparer(t.coords(s), x, comparaisons);
        for (mu, &xm) in x.iter().enumerate() {
            let haut = g.nb_up(s, mu, xm);
            let bas = g.nb_dn(s, mu, xm);
            ecarts += comparer(t.voisin_haut(s, mu), haut, comparaisons);
            ecarts += comparer(t.voisin_bas(s, mu), bas, comparaisons);
            ecarts += comparer(t.lien(mu, s), mu * g.v + s, comparaisons);
            ecarts += comparer(ligne[mu], haut as u32, comparaisons);
            ecarts += comparer(ligne[mu + 4], bas as u32, comparaisons);
            ecarts += comparer(t.gele(t.lien(mu, s)), false, comparaisons);
        }
        for p in 0..6 {
            ecarts += comparer(t.coef(p, s), 1.0, comparaisons);
        }
    }
    for p in 0..2 {
        let classe = t.classes.get(p);
        ecarts += comparer(classe.map(Vec::len), Some(g.v / 2), comparaisons);
        for i in 0..g.v / 2 {
            ecarts += comparer(
                classe.and_then(|c| c.get(i)).copied(),
                Some(g.site_of(i, p).0),
                comparaisons,
            );
        }
    }
    ecarts
}

fn site_c(x: &[usize; 4], cotes: &[usize; 4]) -> usize {
    ((x[0] * cotes[1] + x[1]) * cotes[2] + x[2]) * cotes[3] + x[3]
}

/// Référence indépendante des pas : boucles imbriquées dans l'ordre C et voisins par coordonnées.
fn ecarts_ordre_c(t: &Tables, comparaisons: &mut usize) -> usize {
    let mut ecarts = 0;
    let mut compteur = 0;
    for x0 in 0..t.cotes[0] {
        for x1 in 0..t.cotes[1] {
            for x2 in 0..t.cotes[2] {
                for x3 in 0..t.cotes[3] {
                    let x = [x0, x1, x2, x3];
                    ecarts += comparer(t.site(&x), compteur, comparaisons);
                    ecarts += comparer(t.coords(compteur), x, comparaisons);
                    for (mu, &cote) in t.cotes.iter().enumerate() {
                        let mut haut = x;
                        let mut bas = x;
                        haut[mu] = (x[mu] + 1) % cote;
                        bas[mu] = (x[mu] + cote - 1) % cote;
                        ecarts += comparer(
                            t.voisin_haut(compteur, mu),
                            site_c(&haut, &t.cotes),
                            comparaisons,
                        );
                        ecarts += comparer(
                            t.voisin_bas(compteur, mu),
                            site_c(&bas, &t.cotes),
                            comparaisons,
                        );
                    }
                    compteur += 1;
                }
            }
        }
    }
    ecarts += comparer(compteur, t.v, comparaisons);
    ecarts
}

fn ecarts_torsion(t: &Tables, tor: &Torsion, l: usize, comparaisons: &mut usize) -> usize {
    let g = Geo::new(l);
    let mut ecarts = 0;
    for (p, &(mu, nu)) in PLANS.iter().enumerate() {
        for s in 0..g.v {
            ecarts += comparer(t.coef(p, s), tor.z(&g.coords(s), mu, nu, l), comparaisons);
        }
    }
    ecarts
}

fn ecarts_coloriage(t: &Tables, comparaisons: &mut usize) -> usize {
    let mut couleurs = vec![usize::MAX; t.v];
    let mut ecarts = 0;
    for (couleur, classe) in t.classes.iter().enumerate() {
        for &s in classe {
            if let Some(c) = couleurs.get_mut(s) {
                ecarts += comparer(*c, usize::MAX, comparaisons);
                *c = couleur;
            } else {
                ecarts += comparer(s < t.v, true, comparaisons);
            }
        }
    }
    for (s, &couleur) in couleurs.iter().enumerate() {
        ecarts += comparer(couleur != usize::MAX, true, comparaisons);
        for mu in 0..4 {
            for voisin in [t.voisin_haut(s, mu), t.voisin_bas(s, mu)] {
                let distincte = couleurs
                    .get(voisin)
                    .is_some_and(|&c| c != usize::MAX && c != couleur);
                ecarts += comparer(distincte, true, comparaisons);
            }
        }
    }
    ecarts
}

fn ecarts_masque(t: &Tables, attendu: &[u8], comparaisons: &mut usize) -> usize {
    let mut ecarts = comparer(t.masque.is_some(), true, comparaisons);
    if let Some(masque) = &t.masque {
        ecarts += comparer(masque.len(), attendu.len(), comparaisons);
        for (li, (&gele, &reference)) in masque.iter().zip(attendu).enumerate() {
            ecarts += comparer(gele, reference, comparaisons);
            ecarts += comparer(t.gele(li), reference != 0, comparaisons);
        }
    }
    ecarts
}

fn ecarts_refus<T>(resultat: Result<T, Erreur>, comparaisons: &mut usize) -> usize {
    comparer(
        resultat.err().map(|e| e.code_sortie()),
        Some(2),
        comparaisons,
    )
}

#[test]
fn g_c0() {
    let mut n = 0;
    let mut ecarts = 0;
    for l in [4, 6, 8] {
        let spec = GeoSpec::cubique(l, 3);
        let tables = Tables::compiler(&spec).unwrap();
        ecarts += comparer(spec.chemin(), Chemin::C0, &mut n);
        ecarts += ecarts_c0(&tables, l, &mut n);
        ecarts += ecarts_ordre_c(&tables, &mut n);
        ecarts += ecarts_coloriage(&tables, &mut n);
    }
    terminer("G-C0", ecarts, n);
}

#[test]
fn g_ordre_c_et_coloriage() {
    let mut n = 0;
    let mut ecarts = 0;
    for cotes in [[2, 3, 4, 5], [4, 4, 2, 2], [3, 3, 3, 3], [4, 6, 3, 2]] {
        let spec = GeoSpec::anisotrope(cotes, 3);
        let tables = Tables::compiler(&spec).unwrap();
        ecarts += ecarts_ordre_c(&tables, &mut n);
        ecarts += ecarts_coloriage(&tables, &mut n);
        let pair = cotes.iter().all(|c| c.is_multiple_of(2));
        ecarts += comparer(
            spec.coloriage(),
            if pair { "parite" } else { "glouton" },
            &mut n,
        );
        ecarts += comparer(
            tables.coloriage,
            if pair {
                Coloriage::Parite
            } else {
                Coloriage::Glouton
            },
            &mut n,
        );
        ecarts += comparer(spec.chemin(), Chemin::C1, &mut n);
    }
    terminer("G-ORDRE-C-COLORIAGE", ecarts, n);
}

#[test]
fn g_torsion() {
    let mut n = 0;
    let mut ecarts = 0;
    let beta = [1.0, 2.0, 0.5, 3.0, 0.25, 1.5];
    for l in [4, 6] {
        for texte in ["01", "23", "01,23", "02,13", "01,02,03,12,13,23"] {
            let tor = Torsion::analyser(texte).unwrap();
            let spec = GeoSpec::cubique(l, 3).avec_torsion(tor);
            let tables = Tables::compiler(&spec).unwrap();
            ecarts += comparer(spec.chemin(), Chemin::C1, &mut n);
            ecarts += comparer(tables.coef.is_some(), true, &mut n);
            ecarts += ecarts_torsion(&tables, &tor, l, &mut n);
            let tables = Tables::compiler(&spec.avec_beta_plans(beta)).unwrap();
            let g = Geo::new(l);
            for (p, &(mu, nu)) in PLANS.iter().enumerate() {
                for s in 0..g.v {
                    ecarts += comparer(
                        tables.coef(p, s),
                        beta[p] * tor.z(&g.coords(s), mu, nu, l),
                        &mut n,
                    );
                }
            }
        }
    }
    let spec = GeoSpec::cubique(4, 0).avec_beta_plans([1.0; 6]);
    ecarts += comparer(spec.chemin(), Chemin::C1, &mut n);
    let tables = Tables::compiler(&spec).unwrap();
    ecarts += comparer(tables.coef.is_some(), true, &mut n);
    ecarts += ecarts_torsion(&tables, &Torsion::triviale(), 4, &mut n);
    terminer("G-TORSION", ecarts, n);
}

#[test]
fn g_masque() {
    let mut n = 0;
    let mut ecarts = 0;
    for l in [4, 6] {
        let spec = GeoSpec::hyperplan(l, 3, SourceBord::Froid);
        let tables = Tables::compiler(&spec).unwrap();
        ecarts += ecarts_masque(&tables, &masque_gel_x0_spatial(l), &mut n);
        ecarts += comparer(spec.chemin(), Chemin::C1, &mut n);
    }
    for ell in [2, 4] {
        let spec = GeoSpec::boite(ell, 0, SourceBord::Froid);
        let tables = Tables::compiler(&spec).unwrap();
        let reference = masque_boite(&GeometrieBoite::nouvelle(ell).unwrap());
        ecarts += ecarts_masque(&tables, &reference, &mut n);
        ecarts += comparer(spec.chemin(), Chemin::C1, &mut n);
    }
    terminer("G-MASQUE", ecarts, n);
}

#[test]
fn g_json() {
    let specs = [
        GeoSpec::cubique(4, 3),
        GeoSpec::anisotrope([4, 6, 3, 2], 1)
            .avec_torsion(Torsion::analyser("01,23").unwrap())
            .avec_beta_plans([1.0, 2.0, 0.5, 3.0, 0.25, 1.5]),
        GeoSpec::hyperplan(4, 2, SourceBord::Froid),
        GeoSpec::hyperplan(6, 0, SourceBord::Chaud { graine: u64::MAX }),
        GeoSpec::boite(
            2,
            3,
            SourceBord::Npy {
                chemin: "bord.npy".into(),
                sha256: Some("0123456789abcdef".repeat(4)),
            },
        ),
        GeoSpec::repliques(4, 4, 2, 1, 2),
    ];
    let mut n = 0;
    let mut ecarts = 0;
    for spec in specs {
        let bloc = spec.bloc();
        ecarts += comparer(GeoSpec::depuis_json(&bloc).unwrap(), spec.clone(), &mut n);
        let texte = spec.texte();
        let v: Value = serde_json::from_str(&texte).unwrap();
        ecarts += comparer(GeoSpec::depuis_json(&v).unwrap().texte(), texte, &mut n);
        ecarts += comparer(
            kvu::regen::schema::valider(&schema(), &bloc).unwrap().len(),
            0,
            &mut n,
        );
    }
    // Les autres variantes des sources ont le même contrat d'aller-retour.
    for valeurs in [
        SourceBord::Copie {
            source: "chaine/c000.npy".into(),
        },
        SourceBord::ParChaine,
        SourceBord::Npy {
            chemin: "bord.npy".into(),
            sha256: None,
        },
    ] {
        let spec = GeoSpec::hyperplan(4, 3, valeurs);
        ecarts += comparer(GeoSpec::depuis_json(&spec.bloc()).unwrap(), spec, &mut n);
    }
    let copie = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../schemas/kvu-geometrie-1.schema.json"
    ));
    ecarts += comparer(SCHEMA_TEXTE.len(), copie.len(), &mut n);
    for (&a, &b) in SCHEMA_TEXTE.as_bytes().iter().zip(copie.as_bytes()) {
        ecarts += comparer(a, b, &mut n);
    }
    terminer("G-JSON", ecarts, n);
}

#[test]
fn g_refus_spec() {
    let mut mauvais = vec![GeoSpec::anisotrope([4, 1, 4, 4], 3), GeoSpec::cubique(4, 4)];
    let mut direction = GeoSpec::cubique(4, 0);
    direction.bords[2] = Bord::Recolle {
        repliques: 2,
        region: RegionSpec::Tranche { axe: 1, largeur: 2 },
        couture: Couture::Symetrique,
    };
    mauvais.push(direction);
    mauvais.push(GeoSpec::repliques(4, 4, 2, 1, 2).avec_torsion(Torsion::analyser("01").unwrap()));
    mauvais.push(GeoSpec::cubique(4, 3).avec_beta_plans([0.0; 6]));
    mauvais.push(GeoSpec::cubique(4, 3).avec_beta_plans([f64::NAN; 6]));
    let mut boite = GeoSpec::boite(2, 3, SourceBord::Froid);
    boite.cotes[3] = 6;
    mauvais.push(boite);
    mauvais.push(GeoSpec::cubique(usize::MAX, 3));
    mauvais.push(GeoSpec::anisotrope([256; 4], 3));
    mauvais.push(GeoSpec::repliques(4, 4, 1, 1, 2));
    mauvais.push(GeoSpec::repliques(4, 4, 2, 0, 2));
    mauvais.push(GeoSpec::repliques(4, 4, 2, 4, 2));
    mauvais.push(GeoSpec::repliques(4, 4, 2, 1, 5));
    mauvais.push(GeoSpec::boite(3, 3, SourceBord::Froid));
    mauvais.push(GeoSpec::boite(128, 3, SourceBord::Froid));
    mauvais.push(GeoSpec::boite(
        2,
        3,
        SourceBord::Npy {
            chemin: "bord.npy".into(),
            sha256: Some("A".repeat(64)),
        },
    ));
    // O6 : deux bords fixes (valeurs indéfinies à l'intersection des hyperplans) ; O7 : capacités C2 dépassées
    // (repliques(128, 128, 3, …) : 4V > 2³¹ ; repliques(128, 64, 2, …) : 4V = 2³⁰ mais 6V + couture > 2³⁰ − 1 plaquettes)
    let mut deux_fixes = GeoSpec::hyperplan(4, 3, SourceBord::Froid);
    deux_fixes.bords[1] = Bord::Fixe {
        valeurs: SourceBord::Chaud { graine: 1 },
    };
    mauvais.push(deux_fixes);
    mauvais.push(GeoSpec::repliques(128, 128, 3, 1, 0));
    mauvais.push(GeoSpec::repliques(128, 64, 2, 1, 0));
    // N1 (relecture c04b) : union des deux géométries de l'interpolation = 2³⁰ plaquettes, la dernière incidence vaudrait NUL
    mauvais.push(GeoSpec::repliques(2, 11_184_810, 2, 1, 1));
    let mut n = 0;
    let mut ecarts = 0;
    for spec in mauvais {
        ecarts += ecarts_refus(spec.valider(), &mut n);
    }
    // du bon côté : accepté (4V = 2²⁷, union majorée par 2 × (6V + couture) ≈ 4,1·10⁸ plaquettes)
    ecarts += comparer(
        GeoSpec::repliques(64, 64, 2, 1, 0).valider().is_ok(),
        true,
        &mut n,
    );
    let mut periodique = GeoSpec::cubique(4, 3).bloc();
    periodique["bords"][0]["repliques"] = json!(2);
    ecarts += ecarts_refus(GeoSpec::depuis_json(&periodique), &mut n);
    let mut masque = GeoSpec::cubique(4, 3);
    masque.masque = Some(SourceMasque::Boite {
        ell: 4,
        valeurs: SourceBord::Froid,
    });
    ecarts += ecarts_refus(masque.valider(), &mut n);
    terminer("G-REFUS-SPEC", ecarts, n);
}

#[test]
fn g_refus_regles_croisees_json() {
    let mut n = 0;
    let mut ecarts = 0;
    let base = GeoSpec::hyperplan(4, 3, SourceBord::Froid).bloc();
    for valeurs in [
        json!({"type": "froid", "graine": 1}),
        json!({"type": "chaud"}),
        json!({"type": "chaud", "graine": 1, "source": "autre"}),
        json!({"type": "copie"}),
        json!({"type": "copie", "source": "a", "graine": 1}),
        json!({"type": "npy", "chemin": "bord.npy"}),
        json!({"type": "npy", "sha256": null}),
        json!({"type": "npy", "chemin": "bord.npy", "sha256": "0".repeat(63)}),
        json!({"type": "par_chaine", "chemin": "bord.npy"}),
    ] {
        let mut mutant = base.clone();
        mutant["bords"][0]["valeurs"] = valeurs;
        ecarts += ecarts_refus(GeoSpec::depuis_json(&mutant), &mut n);
    }
    for (champ, valeur) in [
        ("cotes", json!([4, 4, 4, 4, 4])),
        ("bords", json!([])),
        ("beta_plans", json!([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
        ("coloriage", json!("glouton")),
        ("disposition", json!("autre disposition")),
    ] {
        let mut mutant = GeoSpec::cubique(4, 3).bloc();
        mutant[champ] = valeur;
        ecarts += ecarts_refus(GeoSpec::depuis_json(&mutant), &mut n);
    }
    let mut mutant = GeoSpec::cubique(4, 3).bloc();
    mutant["torsion"]["plans"] = json!(["01", "01"]);
    ecarts += ecarts_refus(GeoSpec::depuis_json(&mutant), &mut n);
    mutant["torsion"]["plans"] = json!([]);
    mutant["torsion"]["pile"] = json!("mauvais coin");
    ecarts += ecarts_refus(GeoSpec::depuis_json(&mutant), &mut n);
    for champ in ["repliques", "region", "couture"] {
        let mut mutant = GeoSpec::repliques(4, 4, 2, 1, 2).bloc();
        mutant["bords"][0].as_object_mut().unwrap().remove(champ);
        ecarts += ecarts_refus(GeoSpec::depuis_json(&mutant), &mut n);
    }
    let mut mutant = GeoSpec::repliques(4, 4, 2, 1, 2).bloc();
    mutant["bords"][0]["repliques"] = json!(256);
    ecarts += ecarts_refus(GeoSpec::depuis_json(&mutant), &mut n);
    terminer("G-REFUS-JSON", ecarts, n);
}

#[test]
fn mutant_m1_pas_permute() {
    let mut tables = Tables::compiler(&GeoSpec::anisotrope([2, 3, 4, 5], 3)).unwrap();
    let [c0, c1, c2, c3] = tables.cotes;
    let permutes = [c0, c2, c1, c3];
    tables.pas = [
        permutes[1] * permutes[2] * permutes[3],
        permutes[2] * permutes[3],
        permutes[3],
        1,
    ];
    let mut n = 0;
    let ecarts = ecarts_ordre_c(&tables, &mut n);
    terminer_mutant("M1-PAS-PERMUTE", ecarts, n);
}

#[test]
fn mutant_m2_plans_permutes() {
    let tor = Torsion::analyser("01").unwrap();
    let mut tables = Tables::compiler(&GeoSpec::cubique(4, 3).avec_torsion(tor)).unwrap();
    let v = tables.v;
    let coef = tables.coef.as_mut().unwrap();
    let (premier, suite) = coef.split_at_mut(v);
    for (a, b) in premier.iter_mut().zip(&mut suite[..v]) {
        std::mem::swap(a, b);
    }
    let mut n = 0;
    let ecarts = ecarts_torsion(&tables, &tor, 4, &mut n);
    terminer_mutant("M2-PLANS-PERMUTES", ecarts, n);
}

#[test]
fn mutant_m3_dir_temps_absent() {
    let mut v = GeoSpec::cubique(4, 3).bloc();
    v.as_object_mut().unwrap().remove("dir_temps");
    let erreur = GeoSpec::depuis_json(&v).unwrap_err();
    let mut n = 0;
    assert_eq!(comparer(erreur.code_sortie(), 2, &mut n), 0);
    let ecarts = comparer(erreur.code_sortie(), 0, &mut n);
    terminer_mutant("M3-DIR-TEMPS-ABSENT", ecarts, n);
}

#[test]
fn mutant_m4_chemin_faux() {
    let mut v = GeoSpec::cubique(4, 3).bloc();
    v["chemin"] = json!("C1");
    let erreur = GeoSpec::depuis_json(&v).unwrap_err();
    let mut n = 0;
    assert_eq!(comparer(erreur.code_sortie(), 2, &mut n), 0);
    let ecarts = comparer(erreur.code_sortie(), 0, &mut n);
    terminer_mutant("M4-CHEMIN-FAUX", ecarts, n);
}

#[test]
fn mutant_m5_parite_sur_cotes_impairs() {
    let mut tables = Tables::compiler(&GeoSpec::cubique(3, 3)).unwrap();
    tables.classes = vec![Vec::new(), Vec::new()];
    tables.coloriage = Coloriage::Parite;
    for s in 0..tables.v {
        let p = tables.coords(s).iter().sum::<usize>() % 2;
        tables.classes[p].push(s);
    }
    let mut n = 0;
    let ecarts = ecarts_coloriage(&tables, &mut n);
    terminer_mutant("M5-PARITE-IMPAIRE", ecarts, n);
}
