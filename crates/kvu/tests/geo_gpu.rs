//! Portes GPU G-GEO, P2 à P5 et contrôles négatifs du chemin C1 (B10, lot A).
//! Toutes sont ignorées par défaut ; un SKIP ne vaut pas un résultat vert.
//!
//! Sous le verrou GPU de l'acemagic (`/tmp/kevinotron_gpu.lock`) :
//! ```text
//! cargo test -p kvu --release --test geo_gpu -- --ignored --nocapture --test-threads=1
//! ```
//! Sur le gamer, sous `/tmp/gamer_gpu.lock` :
//! ```text
//! cargo test -p kvu --release --features cuda --test geo_gpu -- --ignored --nocapture --test-threads=1
//! ```
//! Le backend suit KEVINOTRON_BACKEND, sinon CUDA avec la feature `cuda`, sinon HIP. Un GPU indisponible est un ÉCHEC
//! (relecture c09, O6) ; KVU_GPU_SAUT=1 demande explicitement de sauter (SKIP, qui ne vaut jamais vert).

use kvu::commun::{code, BackendKind, Depart, Erreur, Graine, Precision, Resultat};
use kvu::geom::{GeoSpec, SourceBord, Tables};
use kvu::gpu::{self, Appareil, OptionsCompil};
use kvu::su2::cpu::Cpu64;
use kvu::su2::geo_gpu::{tableau_geo, tester_geometrie, Geo64Gpu};
use kvu::su2::ref64::Ref64Gpu;
use kvu::su2::torsion::Torsion;
use kvu::su2::{ConfigLot, Gel, NoyauxSu2, ParamChaine};

fn backend_du_test() -> BackendKind {
    match std::env::var("KEVINOTRON_BACKEND")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        Some(b) if b != BackendKind::Cpu => b,
        _ if cfg!(feature = "cuda") => BackendKind::Cuda,
        _ => BackendKind::Hip,
    }
}

fn appareil_ou_skip(nom: &str) -> Option<Appareil> {
    let b = backend_du_test();
    match gpu::ouvrir(b, 0) {
        Ok(app) => Some(app),
        Err(e) => {
            // O6 (relecture c09) : une porte GPU EXÉCUTÉE (--ignored) sans GPU ÉCHOUE par défaut ; le saut n'existe que sur
            // demande explicite (KVU_GPU_SAUT=1) et un SKIP ne vaut jamais vert.
            if std::env::var("KVU_GPU_SAUT").is_ok_and(|v| v == "1") {
                verdict(&format!(
                    "SKIP {nom} : 0 cas exécuté, backend {} indisponible ({e}), saut demandé",
                    b.nom()
                ));
                return None;
            }
            verdict(&format!(
                "FAIL {nom} : backend {} indisponible ({e}) ; KVU_GPU_SAUT=1 pour sauter",
                b.nom()
            ));
            panic!("FAIL {nom} : backend {} indisponible ({e})", b.nom());
        }
    }
}

/// Place le verdict au début d'une ligne, même avec le préfixe de libtest.
fn verdict(ligne: &str) {
    println!("\n{ligne}");
}

fn gate(nom: &str, corps: impl FnOnce() -> Result<String, String>) {
    match corps() {
        Ok(detail) => verdict(&format!("PASS {nom} : {detail}")),
        Err(raison) => {
            verdict(&format!("FAIL {nom} : {raison}"));
            panic!("FAIL {nom} : {raison}");
        }
    }
}

fn lot(l: usize, chaines: &[(f64, u64, Depart)], nor: usize, options: OptionsCompil) -> ConfigLot {
    ConfigLot {
        l,
        chaines: chaines
            .iter()
            .map(|&(beta, g, depart)| ParamChaine {
                beta,
                graine: Graine(g),
                depart,
            })
            .collect(),
        nor,
        precision: Precision::F64,
        options,
        mutant_metro: false,
    }
}

fn bits(q: &[f64]) -> Vec<u64> {
    q.iter().map(|x| x.to_bits()).collect()
}

/// Un écart non fini ou une taille différente ne peut jamais rendre une porte verte.
fn ecart_max(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let e = (x - y).abs();
            if e.is_finite() {
                e
            } else {
                f64::INFINITY
            }
        })
        .fold(0.0, f64::max)
}

fn comparer_exact<T: PartialEq + std::fmt::Debug>(
    nom: &str,
    a: &[T],
    b: &[T],
) -> Result<usize, String> {
    if a.len() != b.len() {
        return Err(format!(
            "{nom} : tailles {} et {}, premier indice absent {}",
            a.len(),
            b.len(),
            a.len().min(b.len())
        ));
    }
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x != y) {
        return Err(format!(
            "{nom} : premier indice différent {i}, {:?} contre {:?}",
            a[i], b[i]
        ));
    }
    Ok(a.len())
}

fn comparer_bits(nom: &str, a: &[f64], b: &[f64]) -> Result<usize, String> {
    comparer_exact(nom, &bits(a), &bits(b))
}

fn comparer_champs(a: &dyn NoyauxSu2, b: &dyn NoyauxSu2, contexte: &str) -> Result<usize, String> {
    if a.config().chaines.len() != b.config().chaines.len() {
        return Err(format!("{contexte} : nombres de chaînes différents"));
    }
    let mut mots = 0;
    for c in 0..a.config().chaines.len() {
        let qa = a.telecharger(c).map_err(|e| e.to_string())?;
        let qb = b.telecharger(c).map_err(|e| e.to_string())?;
        mots += comparer_bits(&format!("{contexte}, chaîne {c}"), &qa.q, &qb.q)?;
    }
    Ok(mots)
}

fn comparer_compteurs(
    a: &dyn NoyauxSu2,
    b: &dyn NoyauxSu2,
    contexte: &str,
) -> Result<usize, String> {
    let ca: Vec<u64> = a
        .compteurs()
        .map_err(|e| e.to_string())?
        .into_iter()
        .flat_map(|c| [c.plafond, c.essais])
        .collect();
    let cb: Vec<u64> = b
        .compteurs()
        .map_err(|e| e.to_string())?
        .into_iter()
        .flat_map(|c| [c.plafond, c.essais])
        .collect();
    comparer_exact(contexte, &ca, &cb)
}

fn avancer(n: &mut dyn NoyauxSu2, pas: usize, nor: usize) -> Result<(), String> {
    for _ in 0..pas {
        n.sweep_hb().map_err(|e| e.to_string())?;
        for _ in 0..nor {
            n.sweep_or().map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn compter_ecarts<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    a.len().abs_diff(b.len()) + a.iter().zip(b).filter(|(x, y)| x != y).count()
}

#[test]
#[ignore = "GPU"]
fn g_geo() {
    let Some(app) = appareil_ou_skip("G-GEO") else {
        return;
    };
    gate("G-GEO", || {
        let mut comparaisons = 0;
        for cotes in [[4; 4], [8; 4], [2, 4, 6, 8], [6, 2, 4, 4]] {
            let t = Tables::compiler(&GeoSpec::anisotrope(cotes, 0)).map_err(|e| e.to_string())?;
            let (voisins, sites) = tester_geometrie(&app, &t, None).map_err(|e| e.to_string())?;
            let attendus: Vec<u32> = t.voisins().into_iter().flatten().collect();
            comparaisons += comparer_exact(&format!("{cotes:?}, voisins"), &voisins, &attendus)?;
            if sites.len() != t.v {
                return Err(format!(
                    "{cotes:?} : {} sites reçus, {} attendus",
                    sites.len(),
                    t.v
                ));
            }
            for (p, classe) in t.classes.iter().enumerate() {
                let attendus: Vec<u32> = classe.iter().map(|&s| s as u32).collect();
                comparaisons += comparer_exact(
                    &format!("{cotes:?}, sites de parité {p}"),
                    &sites[p * (t.v / 2)..(p + 1) * (t.v / 2)],
                    &attendus,
                )?;
            }
        }
        Ok(format!(
            "4 géométries, {comparaisons} comparaisons entières exactes, ordre des parités inclus"
        ))
    });
}

#[test]
#[ignore = "GPU"]
fn g_geo_mutant() {
    let Some(app) = appareil_ou_skip("G-GEO-MUTANT") else {
        return;
    };
    gate("G-GEO-MUTANT", || {
        let mut rouges = 0;
        let mut details = Vec::new();
        for cotes in [[2, 4, 6, 8], [4; 4]] {
            let t = Tables::compiler(&GeoSpec::anisotrope(cotes, 0)).map_err(|e| e.to_string())?;
            let mut geo = tableau_geo(&t);
            geo.swap(6, 7); // pas1 et pas2 échangés.
            let (voisins, sites) =
                tester_geometrie(&app, &t, Some(geo)).map_err(|e| e.to_string())?;
            let attendus: Vec<u32> = t.voisins().into_iter().flatten().collect();
            let classes: Vec<u32> = t.classes.iter().flatten().map(|&s| s as u32).collect();
            let differences =
                compter_ecarts(&voisins, &attendus) + compter_ecarts(&sites, &classes);
            rouges += usize::from(differences > 0);
            details.push(format!("{cotes:?} : {differences} différences"));
        }
        let detail = format!(
            "{rouges}/2 géométries rouges comme attendu, {}",
            details.join(" ; ")
        );
        if rouges == 2 {
            Ok(detail)
        } else {
            Err(detail)
        }
    });
}

#[test]
#[ignore = "GPU"]
fn p2_c1_trivial_egal_c0() {
    let Some(app) = appareil_ou_skip("P2") else {
        return;
    };
    gate("P2", || {
        let mut mots = 0;
        let mut cas = 0;
        for l in [4, 8] {
            for avec_coef in [false, true] {
                let cfg = lot(
                    l,
                    &[(2.3, 1111, Depart::Chaud), (2.5, 2222, Depart::Froid)],
                    4,
                    OptionsCompil::reference_k1(),
                );
                let nor = cfg.nor;
                let mut spec = GeoSpec::cubique(l, 0);
                if avec_coef {
                    spec = spec.avec_beta_plans([1.0; 6]);
                }
                let contexte = format!("L = {l}, coef de 1.0 = {avec_coef}");
                let mut c0 = Ref64Gpu::nouveau(&app, cfg.clone()).map_err(|e| e.to_string())?;
                let mut c1 = Geo64Gpu::nouveau(&app, cfg, &spec).map_err(|e| e.to_string())?;
                c0.init().map_err(|e| e.to_string())?;
                c1.init().map_err(|e| e.to_string())?;
                c0.allouer_historique(21).map_err(|e| e.to_string())?;
                c1.allouer_historique(21).map_err(|e| e.to_string())?;
                c0.mesurer(0).map_err(|e| e.to_string())?;
                c1.mesurer(0).map_err(|e| e.to_string())?;
                mots += comparer_champs(&c0, &c1, &format!("{contexte}, pas 0"))?;
                for pas in 1..=20 {
                    if pas == 1 {
                        let d0 = c0.sweep_hb_diag().map_err(|e| e.to_string())?;
                        let d1 = c1.sweep_hb_diag().map_err(|e| e.to_string())?;
                        comparer_exact(&format!("{contexte}, diagnostics du pas 1"), &d0, &d1)?;
                    } else {
                        c0.sweep_hb().map_err(|e| e.to_string())?;
                        c1.sweep_hb().map_err(|e| e.to_string())?;
                    }
                    for _ in 0..nor {
                        c0.sweep_or().map_err(|e| e.to_string())?;
                        c1.sweep_or().map_err(|e| e.to_string())?;
                    }
                    c0.mesurer(pas).map_err(|e| e.to_string())?;
                    c1.mesurer(pas).map_err(|e| e.to_string())?;
                    if [1, 10, 20].contains(&pas) {
                        mots += comparer_champs(&c0, &c1, &format!("{contexte}, pas {pas}"))?;
                    }
                }
                let h0 = c0.historique().map_err(|e| e.to_string())?;
                let h1 = c1.historique().map_err(|e| e.to_string())?;
                mots += comparer_bits(&format!("{contexte}, historique"), &h0, &h1)?;
                mots += comparer_compteurs(&c0, &c1, &format!("{contexte}, compteurs"))?;
                cas += 1;
            }
        }
        Ok(format!(
            "{cas} cas, 2 chaînes, 20 pas et 4 OR par pas, {mots} mots de 64 bits égaux"
        ))
    });
}

/// P2-COMPAT : en `compat-prod` (contraction FMA autorisée), C1 (spécification C0, coef de 1.0) contre C0 sur la même carte,
/// chaque balayage depuis un état commun (C0 resynchronisé sur C1) : ÉGALITÉ BINAIRE exigée (relecture c09, O7 : observée
/// 100/100 sur HIP et CUDA ; un compilateur qui contracterait autrement les deux textes ferait rougir la porte).
#[test]
#[ignore = "GPU"]
fn p2_compat_prod() {
    let Some(app) = appareil_ou_skip("P2-COMPAT") else {
        return;
    };
    gate("P2-COMPAT", || {
        let (mut divergentes, mut pire, mut balayages, mut identiques) =
            (0usize, 0.0f64, 0usize, 0usize);
        for l in [4, 8] {
            let opts = OptionsCompil::compat_production(app.backend());
            let cfg = lot(
                l,
                &[(2.3, 3331, Depart::Chaud), (2.5, 3332, Depart::Froid)],
                4,
                opts,
            );
            let nor = cfg.nor;
            let spec = GeoSpec::cubique(l, 0).avec_beta_plans([1.0; 6]);
            let mut c0 = Ref64Gpu::nouveau(&app, cfg.clone()).map_err(|e| e.to_string())?;
            let mut c1 = Geo64Gpu::nouveau(&app, cfg, &spec).map_err(|e| e.to_string())?;
            c1.init().map_err(|e| e.to_string())?;
            c0.init().map_err(|e| e.to_string())?;
            let resync = |c0: &mut Ref64Gpu, c1: &Geo64Gpu| -> Result<(), String> {
                for c in 0..2 {
                    let q = c1.telecharger(c).map_err(|e| e.to_string())?;
                    c0.televerser(c, &q).map_err(|e| e.to_string())?;
                }
                c0.fixer_sweep_id(c1.sweep_id());
                Ok(())
            };
            let ecart = |c0: &Ref64Gpu, c1: &Geo64Gpu| -> Result<(f64, bool), String> {
                let (mut e, mut memes) = (0.0f64, true);
                for c in 0..2 {
                    let (a, b) = (
                        c0.telecharger(c).map_err(|e| e.to_string())?,
                        c1.telecharger(c).map_err(|e| e.to_string())?,
                    );
                    e = e.max(ecart_max(&a.q, &b.q));
                    memes &= bits(&a.q) == bits(&b.q);
                }
                Ok((e, memes))
            };
            for _pas in 0..10 {
                resync(&mut c0, &c1)?;
                let d0 = c0.sweep_hb_diag().map_err(|e| e.to_string())?;
                let d1 = c1.sweep_hb_diag().map_err(|e| e.to_string())?;
                divergentes += compter_ecarts(&d0, &d1);
                let (e, memes) = ecart(&c0, &c1)?;
                pire = pire.max(e);
                identiques += usize::from(memes);
                balayages += 1;
                for _ in 0..nor {
                    resync(&mut c0, &c1)?;
                    c0.sweep_or().map_err(|e| e.to_string())?;
                    c1.sweep_or().map_err(|e| e.to_string())?;
                    let (e, memes) = ecart(&c0, &c1)?;
                    pire = pire.max(e);
                    identiques += usize::from(memes);
                    balayages += 1;
                }
            }
        }
        let detail = format!(
            "L = 4 et 8, 2 chaînes, 10 pas : {balayages} balayages, {divergentes} décisions divergentes, pire max|Δq| = {pire:.3e}, \
             {identiques}/{balayages} balayages identiques au bit (égalité BINAIRE exigée, relecture c09 O7)"
        );
        if divergentes == 0 && identiques == balayages && balayages > 0 {
            Ok(detail)
        } else {
            Err(detail)
        }
    });
}

/// Résultat d'un cas P3/P4. Chaque balayage (bain thermique ou surrelaxation) part d'un état COMMUN : le CPU est resynchronisé
/// sur l'état GPU (champ et `sweep_id`) avant chaque balayage, car la dynamique amplifie les écarts d'arrondi (log/cos du GPU
/// contre la libm) : sans resynchronisation, un écart de 1e-15 croît jusqu'à O(1) en quelques dizaines de balayages, chemin C0
/// compris (diagnostic `diagnostic_derive_libre`, information et non porte). Même procédure que G-CPU-GPU (`su2_gpu.rs`), répétée sur 10 pas.
struct Cas {
    /// Décisions divergentes (essais par lien) cumulées sur les 10 bains thermiques.
    divergentes: usize,
    /// Pire écart max|Δq| d'un balayage du pas 1.
    e1: f64,
    /// Pire écart max|Δq| d'un balayage des pas 1 à 10.
    e10: f64,
    /// Pire écart des 5 mesures (plaquette tordue, 4 Polyakov) sur un état identique, pas 1 à 10.
    em: f64,
}

fn resynchroniser(gpu: &Geo64Gpu, cpu: &mut Cpu64) -> Result<(), String> {
    let q = gpu.telecharger(0).map_err(|e| e.to_string())?;
    cpu.televerser(0, &q).map_err(|e| e.to_string())?;
    cpu.fixer_sweep_id(gpu.sweep_id());
    Ok(())
}

fn ecart_champs(gpu: &Geo64Gpu, cpu: &Cpu64) -> Result<f64, String> {
    let qg = gpu.telecharger(0).map_err(|e| e.to_string())?;
    let qc = cpu.telecharger(0).map_err(|e| e.to_string())?;
    Ok(ecart_max(&qg.q, &qc.q))
}

fn cas_tordu(
    app: &Appareil,
    l: usize,
    torsion: &str,
    depart: Depart,
    mutant: u8,
) -> Result<Cas, String> {
    let cfg = lot(
        l,
        &[(2.5, 26111100 + l as u64, depart)],
        4,
        OptionsCompil::reference_k1(),
    );
    let nor = cfg.nor;
    let torsion = Torsion::analyser(torsion).map_err(|e| e.to_string())?;
    let spec = GeoSpec::cubique(l, 0).avec_torsion(torsion);
    let mut gpu =
        Geo64Gpu::nouveau_mutant(app, cfg.clone(), &spec, mutant).map_err(|e| e.to_string())?;
    gpu.init().map_err(|e| e.to_string())?;
    let mut cpu = Cpu64::nouveau(cfg).map_err(|e| e.to_string())?;
    cpu.init().map_err(|e| e.to_string())?;
    cpu.fixer_torsion(&torsion).map_err(|e| e.to_string())?;
    gpu.allouer_historique(1).map_err(|e| e.to_string())?;
    cpu.allouer_historique(1).map_err(|e| e.to_string())?;
    let mut cas = Cas {
        divergentes: 0,
        e1: 0.0,
        e10: 0.0,
        em: 0.0,
    };
    for pas in 1..=10 {
        // P4 : mesures sur un état identique
        resynchroniser(&gpu, &mut cpu)?;
        gpu.mesurer(0).map_err(|e| e.to_string())?;
        cpu.mesurer(0).map_err(|e| e.to_string())?;
        let hg = gpu.historique().map_err(|e| e.to_string())?;
        let hc = cpu.historique().map_err(|e| e.to_string())?;
        if hg.len() != 5 || hc.len() != 5 {
            return Err(format!(
                "mesure tordue : {} et {} valeurs, 5 attendues",
                hg.len(),
                hc.len()
            ));
        }
        cas.em = cas.em.max(ecart_max(&hg, &hc));
        // P3 : bain thermique depuis l'état commun
        let dg = gpu.sweep_hb_diag().map_err(|e| e.to_string())?;
        let dc = cpu.sweep_hb_diag().map_err(|e| e.to_string())?;
        cas.divergentes += compter_ecarts(&dg, &dc);
        let mut pire = ecart_champs(&gpu, &cpu)?;
        // puis chaque surrelaxation depuis un état commun
        for _ in 0..nor {
            resynchroniser(&gpu, &mut cpu)?;
            gpu.sweep_or().map_err(|e| e.to_string())?;
            cpu.sweep_or().map_err(|e| e.to_string())?;
            pire = pire.max(ecart_champs(&gpu, &cpu)?);
        }
        if pas == 1 {
            cas.e1 = pire;
        }
        cas.e10 = cas.e10.max(pire);
    }
    Ok(cas)
}

/// Seuil de P3-TRAJ-1, fixé AVANT la mesure (11/11, 17h20) : environ 10 fois le témoin C0 observé sur un pas libre
/// (`Ref64Gpu` contre `Cpu64` sans torsion : 7,3e-12 à L = 4, 8,7e-12 à L = 8, HIP et CUDA, `diagnostic_derive_libre`).
const SEUIL_TRAJ_1: f64 = 1e-10;

/// Un pas LIBRE (1 bain + `nor` surrelaxations, sans resynchronisation) depuis un état initial commun : (décisions
/// divergentes, max|Δq|). `torsion` None : témoin C0 (`Ref64Gpu` contre `Cpu64`) ; Some : C1 (`Geo64Gpu`, mutant) contre Cpu64 + M4.
fn trajectoire_1pas(
    app: &Appareil,
    l: usize,
    torsion: Option<&str>,
    depart: Depart,
    mutant: u8,
) -> Result<(usize, f64), String> {
    let cfg = lot(
        l,
        &[(2.5, 26111300 + l as u64, depart)],
        4,
        OptionsCompil::reference_k1(),
    );
    let nor = cfg.nor;
    let mut cpu = Cpu64::nouveau(cfg.clone()).map_err(|e| e.to_string())?;
    cpu.init().map_err(|e| e.to_string())?;
    let (mut gpu, dg): (Box<dyn NoyauxSu2>, Vec<u32>) = match torsion {
        Some(t) => {
            let t = Torsion::analyser(t).map_err(|e| e.to_string())?;
            cpu.fixer_torsion(&t).map_err(|e| e.to_string())?;
            let spec = GeoSpec::cubique(l, 0).avec_torsion(t);
            let mut g =
                Geo64Gpu::nouveau_mutant(app, cfg, &spec, mutant).map_err(|e| e.to_string())?;
            g.init().map_err(|e| e.to_string())?;
            cpu.televerser(0, &g.telecharger(0).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            cpu.fixer_sweep_id(g.sweep_id());
            let d = g.sweep_hb_diag().map_err(|e| e.to_string())?;
            (Box::new(g), d)
        }
        None => {
            let mut g = Ref64Gpu::nouveau(app, cfg).map_err(|e| e.to_string())?;
            g.init().map_err(|e| e.to_string())?;
            cpu.televerser(0, &g.telecharger(0).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            cpu.fixer_sweep_id(g.sweep_id());
            let d = g.sweep_hb_diag().map_err(|e| e.to_string())?;
            (Box::new(g), d)
        }
    };
    let dc = cpu.sweep_hb_diag().map_err(|e| e.to_string())?;
    for _ in 0..nor {
        gpu.sweep_or().map_err(|e| e.to_string())?;
        cpu.sweep_or().map_err(|e| e.to_string())?;
    }
    let qg = gpu.telecharger(0).map_err(|e| e.to_string())?;
    let qc = cpu.telecharger(0).map_err(|e| e.to_string())?;
    Ok((compter_ecarts(&dg, &dc), ecart_max(&qg.q, &qc.q)))
}

/// Dérive libre (sans resynchronisation) GPU contre CPU : max|Δq| après 1, 2, 5, 10 pas et nombre de pas après lesquels les
/// compteurs cumulés d'essais diffèrent.
fn derive_libre(
    app: &Appareil,
    l: usize,
    torsion: Option<&str>,
) -> Result<([f64; 4], usize), String> {
    let cfg = lot(
        l,
        &[(2.5, 26111200 + l as u64, Depart::Chaud)],
        4,
        OptionsCompil::reference_k1(),
    );
    let nor = cfg.nor;
    let mut gpu: Box<dyn NoyauxSu2> = match torsion {
        Some(t) => {
            let t = Torsion::analyser(t).map_err(|e| e.to_string())?;
            Box::new(
                Geo64Gpu::nouveau(app, cfg.clone(), &GeoSpec::cubique(l, 0).avec_torsion(t))
                    .map_err(|e| e.to_string())?,
            )
        }
        None => Box::new(Ref64Gpu::nouveau(app, cfg.clone()).map_err(|e| e.to_string())?),
    };
    gpu.init().map_err(|e| e.to_string())?;
    let mut cpu = Cpu64::nouveau(cfg).map_err(|e| e.to_string())?;
    cpu.init().map_err(|e| e.to_string())?;
    if let Some(t) = torsion {
        cpu.fixer_torsion(&Torsion::analyser(t).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    }
    cpu.televerser(0, &gpu.telecharger(0).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    cpu.fixer_sweep_id(gpu.sweep_id());
    let (mut e, mut divergentes) = ([0.0f64; 4], 0usize);
    for pas in 1..=10 {
        gpu.sweep_hb().map_err(|e| e.to_string())?;
        cpu.sweep_hb().map_err(|e| e.to_string())?;
        for _ in 0..nor {
            gpu.sweep_or().map_err(|e| e.to_string())?;
            cpu.sweep_or().map_err(|e| e.to_string())?;
        }
        if let Some(i) = [1, 2, 5, 10].iter().position(|&p| p == pas) {
            let qg = gpu.telecharger(0).map_err(|e| e.to_string())?;
            let qc = cpu.telecharger(0).map_err(|e| e.to_string())?;
            e[i] = ecart_max(&qg.q, &qc.q);
        }
        let cg = gpu.compteurs().map_err(|e| e.to_string())?;
        let cc = cpu.compteurs().map_err(|e| e.to_string())?;
        divergentes += usize::from(cg[0].essais != cc[0].essais);
    }
    Ok((e, divergentes))
}

#[test]
#[ignore = "GPU"]
fn diagnostic_derive_libre() {
    // INFORMATION, PAS une porte (revue c09b, point 6) : ses chiffres ne sont jugés par aucun seuil ; une ERREUR d'exécution
    // fait en revanche échouer le test (un « ok » veut dire que le diagnostic a bien tourné).
    let Some(app) = appareil_ou_skip("DIAGNOSTIC-DERIVE-LIBRE") else {
        return;
    };
    for l in [4, 8] {
        for torsion in [None, Some("01,23")] {
            let chemin = if torsion.is_some() {
                "C1 tordu 01,23 contre Cpu64+M4"
            } else {
                "C0 Ref64Gpu contre Cpu64 (témoin)"
            };
            match derive_libre(&app, l, torsion) {
                Ok((e, d)) => verdict(&format!(
                    "INFO DIAGNOSTIC-DERIVE-LIBRE L = {l}, {chemin} : max|Δq| après 1/2/5/10 pas = {:.2e} / {:.2e} / {:.2e} / \
                     {:.2e} ; pas avec compteurs cumulés d'essais différents : {d}/10",
                    e[0], e[1], e[2], e[3]
                )),
                Err(err) => {
                    verdict(&format!("FAIL DIAGNOSTIC-DERIVE-LIBRE L = {l}, {chemin} : erreur d'exécution {err}"));
                    panic!("diagnostic de dérive libre en erreur : {err}");
                }
            }
        }
    }
}

#[test]
#[ignore = "GPU"]
fn p3_p4_torsion_gpu_cpu() {
    let Some(app) = appareil_ou_skip("P3") else {
        verdict("SKIP P4 : 0/18 cas exécuté, même appareil indisponible");
        return;
    };
    let mut cas = 0;
    let mut divergentes = 0;
    let (mut e1, mut e10, mut em) = (0.0f64, 0.0f64, 0.0f64);
    let mut erreurs = Vec::new();
    let (mut traj_div, mut traj_e, mut temoin_div, mut temoin_e) = (0usize, 0.0f64, 0usize, 0.0f64);
    for l in [4, 6, 8] {
        for depart in [Depart::Chaud, Depart::Froid] {
            match trajectoire_1pas(&app, l, None, depart, 0) {
                Ok((d, e)) => {
                    temoin_div += d;
                    temoin_e = temoin_e.max(e);
                }
                Err(e) => erreurs.push(format!("témoin C0 L = {l}, départ {depart:?} : {e}")),
            }
        }
        for torsion in ["01", "23", "01,23"] {
            for depart in [Depart::Chaud, Depart::Froid] {
                match trajectoire_1pas(&app, l, Some(torsion), depart, 0) {
                    Ok((d, e)) => {
                        traj_div += d;
                        traj_e = traj_e.max(e);
                    }
                    Err(e) => erreurs.push(format!(
                        "trajectoire L = {l}, torsion {torsion}, départ {depart:?} : {e}"
                    )),
                }
                match cas_tordu(&app, l, torsion, depart, 0) {
                    Ok(c) => {
                        cas += 1;
                        divergentes += c.divergentes;
                        e1 = e1.max(c.e1);
                        e10 = e10.max(c.e10);
                        em = em.max(c.em);
                    }
                    Err(e) => erreurs.push(format!(
                        "L = {l}, torsion {torsion}, départ {depart:?} : {e}"
                    )),
                }
            }
        }
    }
    let p3 = erreurs.is_empty() && cas == 18 && divergentes == 0 && e1 <= 1e-12 && e10 <= 1e-12;
    let p4 = erreurs.is_empty() && cas == 18 && em <= 1e-12;
    let ptraj = erreurs.is_empty() && cas == 18 && traj_div == 0 && traj_e <= SEUIL_TRAJ_1;
    let vt = if ptraj { "PASS" } else { "FAIL" };
    verdict(&format!(
        "{vt} P3-TRAJ-1 : 18 cas, 1 pas LIBRE (1 bain + 4 surrelaxations) depuis un état commun : {traj_div} décisions \
         divergentes, pire max|Δq| = {traj_e:.3e} ≤ {SEUIL_TRAJ_1:.0e} ; témoin C0 (6 cas, sans torsion) : {temoin_div} \
         décisions divergentes, pire {temoin_e:.3e}"
    ));
    let v3 = if p3 { "PASS" } else { "FAIL" };
    let v4 = if p4 { "PASS" } else { "FAIL" };
    verdict(&format!(
        "{v3} P3 : {cas}/18 cas × 10 pas, chaque balayage depuis un état commun : {divergentes} décisions divergentes, \
         pire balayage du pas 1 = {e1:.3e}, des pas 1 à 10 = {e10:.3e}, {} erreurs",
        erreurs.len()
    ));
    verdict(&format!(
        "{v4} P4 : {cas}/18 cas × 10 états identiques, pire écart des 5 mesures = {em:.3e}, {} erreurs",
        erreurs.len()
    ));
    assert!(
        p3 && p4 && ptraj,
        "P3/P4 : au moins une porte rouge ; {}",
        erreurs.join(" ; ")
    );
}

#[test]
#[ignore = "GPU"]
fn p3_p4_mutants() {
    let Some(app) = appareil_ou_skip("P3-MUTANT-1") else {
        for nom in ["P3-MUTANT-2", "P3-MUTANT-3", "P4-MUTANT-4", "P3-MUTANT-5"] {
            verdict(&format!(
                "SKIP {nom} : 0/1 cas exécuté, même appareil indisponible"
            ));
        }
        return;
    };
    let mut tous_rouges = true;
    for mutant in 1..=5 {
        let nom = if mutant == 4 {
            "P4-MUTANT-4".to_string()
        } else {
            format!("P3-MUTANT-{mutant}")
        };
        let traj = trajectoire_1pas(&app, 4, Some("01,23"), Depart::Chaud, mutant);
        let cas = cas_tordu(&app, 4, "01,23", Depart::Chaud, mutant);
        match (verdict_mutant(mutant, &cas, &traj), &cas) {
            (Ok(rouge), Ok(c)) => {
                tous_rouges &= rouge;
                let v = if rouge { "PASS" } else { "FAIL" };
                verdict(&format!(
                    "{v} {nom} : 1 cas terminé, {} décisions divergentes, e1 = {:.3e}, e10 = {:.3e}, em = {:.3e} ; pas libre : {traj:?}",
                    c.divergentes, c.e1, c.e10, c.em
                ));
            }
            (r, c) => {
                let e = r
                    .err()
                    .or_else(|| c.as_ref().err().cloned())
                    .unwrap_or_default();
                tous_rouges = false;
                verdict(&format!("FAIL {nom} : contrôle négatif incomplet ({e})"));
            }
        }
    }
    assert!(tous_rouges, "un mutant C1 n'est pas rouge comme attendu, ou son exécution n'est pas terminée à valeurs finies");
}

/// Revue c09b, point 5 : verdict d'un contrôle négatif. Un mutant n'est « rouge comme attendu » que si ses deux exécutions
/// (`cas_tordu` et le pas libre) se sont TERMINÉES avec des écarts FINIS montrant l'écart attendu ; une erreur d'exécution ou
/// une valeur non finie rend `Err` (échec du contrôle), jamais un faux « rouge ».
fn verdict_mutant(
    mutant: u8,
    cas: &Result<Cas, String>,
    traj: &Result<(usize, f64), String>,
) -> Result<bool, String> {
    let c = cas
        .as_ref()
        .map_err(|e| format!("cas_tordu en erreur : {e}"))?;
    let &(td, te) = traj
        .as_ref()
        .map_err(|e| format!("pas libre en erreur : {e}"))?;
    for (nom, x) in [
        ("e1", c.e1),
        ("e10", c.e10),
        ("em", c.em),
        ("écart du pas libre", te),
    ] {
        if !x.is_finite() {
            return Err(format!("{nom} non fini ({x})"));
        }
    }
    Ok(if mutant == 4 {
        c.em > 1e-12
    } else {
        (c.divergentes > 0 || c.e1 > 1e-12 || c.e10 > 1e-12) && (td > 0 || te > SEUIL_TRAJ_1)
    })
}

/// Revue c09b, point 5 (CPU, non ignoré) : sur 30a35bb, un pas libre en ERREUR donnait NaN et `te.is_nan()` comptait le
/// mutant « rouge » ; désormais une exécution incomplète ou non finie fait échouer le contrôle négatif.
#[test]
fn c09c_p5_controle_negatif_exige_une_trajectoire_terminee() {
    let rouge = || {
        Ok(Cas {
            divergentes: 5,
            e1: 1.9,
            e10: 1.9,
            em: 0.0,
        })
    };
    assert_eq!(verdict_mutant(1, &rouge(), &Ok((3, 1.8))), Ok(true));
    assert!(
        verdict_mutant(1, &rouge(), &Err("panne GPU".into())).is_err(),
        "pas libre en erreur accepté"
    );
    assert!(
        verdict_mutant(1, &rouge(), &Ok((0, f64::NAN))).is_err(),
        "pas libre NaN accepté"
    );
    let infini = Ok(Cas {
        divergentes: 0,
        e1: f64::INFINITY,
        e10: 0.0,
        em: 0.0,
    });
    assert!(
        verdict_mutant(2, &infini, &Ok((1, 1.0))).is_err(),
        "écart non fini accepté"
    );
    assert!(
        verdict_mutant(3, &Err("panne".into()), &Ok((1, 1.0))).is_err(),
        "cas en erreur accepté"
    );
    let vert = Ok(Cas {
        divergentes: 0,
        e1: 2e-15,
        e10: 2e-15,
        em: 0.0,
    });
    assert_eq!(
        verdict_mutant(3, &vert, &Ok((0, 1e-12))),
        Ok(false),
        "mutant vert compté rouge"
    );
    assert_eq!(verdict_mutant(4, &vert, &Ok((0, 1e-12))), Ok(false));
}

#[test]
#[ignore = "GPU"]
fn p5_determinisme() {
    let Some(app) = appareil_ou_skip("P5") else {
        return;
    };
    gate("P5", || {
        let torsion = Torsion::analyser("01,23").map_err(|e| e.to_string())?;
        let spec = GeoSpec::cubique(8, 0).avec_torsion(torsion);
        let cfg = lot(
            8,
            &[(2.3, 51, Depart::Chaud), (2.6, 52, Depart::Froid)],
            4,
            OptionsCompil::reference_k1(),
        );
        let nor = cfg.nor;
        let mut a = Geo64Gpu::nouveau(&app, cfg.clone(), &spec).map_err(|e| e.to_string())?;
        let mut b = Geo64Gpu::nouveau(&app, cfg, &spec).map_err(|e| e.to_string())?;
        a.init().map_err(|e| e.to_string())?;
        b.init().map_err(|e| e.to_string())?;
        a.allouer_historique(21).map_err(|e| e.to_string())?;
        b.allouer_historique(21).map_err(|e| e.to_string())?;
        a.mesurer(0).map_err(|e| e.to_string())?;
        b.mesurer(0).map_err(|e| e.to_string())?;
        for pas in 1..=20 {
            avancer(&mut a, 1, nor)?;
            avancer(&mut b, 1, nor)?;
            a.mesurer(pas).map_err(|e| e.to_string())?;
            b.mesurer(pas).map_err(|e| e.to_string())?;
        }
        let mut mots = comparer_champs(&a, &b, "P5, champs finaux")?;
        let ha = a.historique().map_err(|e| e.to_string())?;
        let hb = b.historique().map_err(|e| e.to_string())?;
        mots += comparer_bits("P5, historiques", &ha, &hb)?;
        mots += comparer_compteurs(&a, &b, "P5, compteurs")?;
        Ok(format!(
            "L = 8, torsion 01,23, 2 chaînes, 20 pas, {mots} mots de 64 bits égaux"
        ))
    });
}

fn verifier_refus<T>(nom: &str, resultat: Resultat<T>, attendu: i32) -> Result<(), String> {
    match resultat {
        Err(e) => {
            let bonne_variante = matches!((&e, attendu), (Erreur::Config(_), code::USAGE));
            if bonne_variante && e.code_sortie() == attendu {
                Ok(())
            } else {
                Err(format!(
                    "{nom} : code {}, attendu {attendu}, erreur {e}",
                    e.code_sortie()
                ))
            }
        }
        Ok(_) => Err(format!("{nom} : accepté, refus de code {attendu} attendu")),
    }
}

#[test]
#[ignore = "GPU"]
fn refus_lot_a() {
    let Some(app) = appareil_ou_skip("REFUS-LOT-A") else {
        return;
    };
    gate("REFUS-LOT-A", || {
        let cfg = lot(
            4,
            &[(2.5, 41, Depart::Chaud)],
            4,
            OptionsCompil::reference_k1(),
        );
        let mut refus = 0;
        for (nom, spec) in [
            ("côtés inégaux", GeoSpec::anisotrope([4, 4, 4, 6], 0)),
            (
                "β par plan",
                GeoSpec::cubique(4, 0).avec_beta_plans([1.0, 2.0, 1.0, 1.0, 1.0, 1.0]),
            ),
            (
                "bord fixé + torsion (lot C : sans référence CPU)",
                GeoSpec::hyperplan(4, 0, SourceBord::Froid)
                    .avec_torsion(Torsion::analyser("01").map_err(|e| e.to_string())?),
            ),
            // relecture Sol : refus AVANT Tables::compiler (sinon 6·180⁴ coefficients ≈ 50 Go alloués sur l'hôte)
            (
                "côtés 180⁴ tordus contre L = 4",
                GeoSpec::cubique(180, 0)
                    .avec_torsion(Torsion::analyser("01").map_err(|e| e.to_string())?),
            ),
        ] {
            verifier_refus(
                nom,
                Geo64Gpu::nouveau(&app, cfg.clone(), &spec),
                code::USAGE,
            )?;
            refus += 1;
        }
        let spec = GeoSpec::cubique(4, 0);
        let mut cfg_f32 = cfg.clone();
        cfg_f32.precision = Precision::F32;
        verifier_refus(
            "précision F32",
            Geo64Gpu::nouveau(&app, cfg_f32, &spec),
            code::USAGE,
        )?;
        refus += 1;
        let torsion = Torsion::analyser("01").map_err(|e| e.to_string())?;
        let spec = spec.avec_torsion(torsion);
        let mut cfg_zero = cfg.clone();
        cfg_zero.chaines[0].beta = 0.0;
        verifier_refus(
            "torsion à β = 0",
            Geo64Gpu::nouveau(&app, cfg_zero, &spec),
            code::USAGE,
        )?;
        refus += 1;
        let mut g = Geo64Gpu::nouveau(&app, cfg, &spec).map_err(|e| e.to_string())?;
        verifier_refus(
            "torsion différente",
            g.fixer_torsion(&Torsion::triviale()),
            code::USAGE,
        )?;
        refus += 1;
        verifier_refus("tranches R2-3", g.mesurer_tranches(0, 1), code::USAGE)?;
        refus += 1;
        verifier_refus(
            "gel du bord avec torsion",
            g.fixer_gel(Gel::BordX0Spatial),
            code::USAGE,
        )?;
        refus += 1;
        Ok(format!(
            "{refus} refus vérifiés, tous Erreur::Config (code {})",
            code::USAGE
        ))
    });
}

const BIN_KVU: &str = env!("CARGO_BIN_EXE_kvu");

fn gen_cli(backend: &str, dir: &std::path::Path) -> std::process::Output {
    let d = dir.to_string_lossy().into_owned();
    std::process::Command::new(BIN_KVU)
        .args([
            "su2",
            "gen",
            "--backend",
            backend,
            "--torsion",
            "01,23",
            "--L",
            "8",
            "--beta",
            "2.5",
            "--graine",
        ])
        .args([
            "26111500",
            "--start",
            "hot",
            "--therm",
            "5",
            "--nmeas",
            "10",
            "--nor",
            "4",
            "--dump-every",
            "5",
        ])
        .args(["--format", "q64", "--outdir", d.as_str()])
        .output()
        .expect("échec de l'exécution de kvu")
}

/// CLI-TORSION-GPU (B10, levée du refus approuvée par le chef d'orchestre le 11/11 à 17h28) : `su2 gen --backend <hip|cuda>
/// --torsion 01,23` passe par le chemin C1 (module `su2_geo64` dans le manifeste), écrit `torsion.json` et COMPLETE, et deux
/// lancements identiques donnent des `measurements.csv` identiques à l'octet. Sans GPU : code 3 (GPU indisponible), jamais un
/// faux refus de torsion (code 2).
#[test]
#[ignore = "GPU"]
fn cli_torsion_gpu() {
    let b = backend_du_test();
    let base = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("geo_gpu")
        .join(format!("cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).expect("dossier de test");
    let Some(app) = appareil_ou_skip("CLI-TORSION-GPU") else {
        let out = gen_cli(b.nom(), &base.join("sans_gpu"));
        verdict(&format!(
            "INFO CLI-TORSION-GPU sans GPU : rc = {:?} (attendu 3), stderr = {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
        assert_eq!(
            out.status.code(),
            Some(3),
            "sans GPU, --backend {} --torsion doit rendre 3",
            b.nom()
        );
        return;
    };
    drop(app);
    gate("CLI-TORSION-GPU", || {
        let mut mesures = Vec::new();
        for essai in ["a", "b"] {
            let d = base.join(essai);
            let out = gen_cli(b.nom(), &d);
            if out.status.code() != Some(0) {
                return Err(format!(
                    "essai {essai} : rc = {:?}, stderr = {}",
                    out.status.code(),
                    String::from_utf8_lossy(&out.stderr)
                ));
            }
            for f in ["torsion.json", "COMPLETE", "measurements.csv"] {
                if !d.join(f).is_file() {
                    return Err(format!("essai {essai} : {f} absent"));
                }
            }
            let man =
                std::fs::read_to_string(d.join("manifest.json")).map_err(|e| e.to_string())?;
            if !man.contains("su2_geo64") {
                return Err(format!(
                    "essai {essai} : module su2_geo64 absent du manifeste"
                ));
            }
            verifier_sorties_c1(&d).map_err(|e| format!("essai {essai} : {e}"))?;
            mesures.push(std::fs::read(d.join("measurements.csv")).map_err(|e| e.to_string())?);
        }
        if mesures[0] != mesures[1] {
            return Err("measurements.csv diffèrent entre deux lancements identiques".into());
        }
        let lignes = String::from_utf8_lossy(&mesures[0]).lines().count();
        let _ = std::fs::remove_dir_all(&base);
        Ok(format!(
            "backend {}, L = 8, torsion 01,23 : rc 0 ×2, torsion.json + COMPLETE + su2_geo64 au manifeste, \
             géométrie C1 écrite (geometrie.json = bloc du manifeste, temps = axe 3, Geo64Gpu, noyaux_c1) et dumps relus \
             par lire_champ_geo, measurements.csv identiques à l'octet ({lignes} lignes)",
            b.nom()
        ))
    });
}

/// O2 (relecture c09) : un run tordu GPU écrit sa géométrie C1 par les fonctions communes du socle et ses dumps se relisent
/// par `stockage::npy::lire_champ_geo` (qui REFUSE un dump tordu sans géométrie canonique).
fn verifier_sorties_c1(d: &std::path::Path) -> Result<(), String> {
    let attendu =
        kvu::su2::geo_gpu::spec_su2_gen(8, Torsion::analyser("01,23").map_err(|e| e.to_string())?);
    let g = kvu::geom::lire_geometrie(&d.join("geometrie.json"))
        .map_err(|e| format!("geometrie.json : {e}"))?;
    if g != attendu {
        return Err(format!(
            "geometrie.json = {} au lieu de {}",
            g.texte(),
            attendu.texte()
        ));
    }
    let man: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(d.join("manifest.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let bloc = kvu::geom::GeoSpec::depuis_json(&man["geometrie"])
        .map_err(|e| format!("bloc geometrie : {e}"))?;
    if bloc != attendu || man["geometrie"]["dir_temps"] != serde_json::json!(3) {
        return Err(format!(
            "bloc geometrie du manifeste incorrect : {}",
            man["geometrie"]
        ));
    }
    let texte = kvu::su2::geo_gpu::SU2_GEO.contenu;
    if man["noyaux_c1"][0]["texte"].as_str() != Some(texte)
        || man["noyaux_c1"][0]["sha256"].as_str()
            != Some(kvu::heritage::manifest::sha256_hex(texte.as_bytes()).as_str())
    {
        return Err(
            "texte source C1 (su2_geo.cu) absent du manifeste ou d'empreinte incohérente".into(),
        );
    }
    if man["moteur_su2"].as_str() != Some(kvu::su2::geo_gpu::MOTEUR_SU2_C1)
        || man["noyaux_c1"][0]["nom"] != "su2_geo.cu"
    {
        return Err(format!(
            "moteur_su2 = {}, noyaux_c1 = {}",
            man["moteur_su2"], man["noyaux_c1"]
        ));
    }
    let mut dumps: Vec<_> = std::fs::read_dir(d)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("cfg_"))
                && p.extension().is_some_and(|x| x == "npy")
        })
        .collect();
    dumps.sort();
    if dumps.is_empty() {
        return Err("aucun dump cfg_*.npy".into());
    }
    for p in &dumps {
        let c = kvu::stockage::npy::lire_champ_geo(p, 3)
            .map_err(|e| format!("lire_champ_geo({}) : {e}", p.display()))?;
        if c.geo != attendu || c.q.len() != 16 * 8usize.pow(4) {
            return Err(format!(
                "{} : géométrie ou taille relues incorrectes",
                p.display()
            ));
        }
    }
    Ok(())
}

fn su2_cli(args: &[&str]) -> std::process::Output {
    std::process::Command::new(BIN_KVU)
        .arg("su2")
        .args(args)
        .output()
        .expect("échec de l'exécution de kvu")
}

fn fichiers(d: &std::path::Path) -> std::collections::BTreeMap<std::path::PathBuf, Vec<u8>> {
    let mut m = std::collections::BTreeMap::new();
    let mut pile = vec![d.to_path_buf()];
    while let Some(x) = pile.pop() {
        for e in std::fs::read_dir(&x).expect("lecture du dossier").flatten() {
            let p = e.path();
            if p.is_dir() {
                pile.push(p);
            } else {
                m.insert(
                    p.strip_prefix(d).expect("préfixe").to_path_buf(),
                    std::fs::read(&p).expect("lecture"),
                );
            }
        }
    }
    m
}

/// O1 (relecture c09) : (a) CLI — un run tordu GPU interrompu consigne « Geo64Gpu/<backend> » ; sa reprise sur CPU est
/// refusée (code 2, rien d'écrit) ; un point de reprise sans invariants n'est jamais repris par le chemin C1 ; la reprise
/// sur le même backend termine le run avec des mesures égales à l'octet à un run non interrompu. (b) moteur — le gel, le
/// masque externe et le moteur sont consignés : un moteur reconstruit sans le gel, avec un autre masque, ou CPU est refusé
/// (code 2) ; avec le gel reposé, la reprise est égale à l'octet au run non interrompu.
#[test]
#[ignore = "GPU"]
fn o1_reprise_gpu_et_invariants() {
    let Some(app) = appareil_ou_skip("O1-REPRISE-GPU") else {
        return;
    };
    let b = backend_du_test().nom();
    gate("O1-REPRISE-GPU", || {
        let base = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join("geo_gpu")
            .join(format!("o1-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).map_err(|e| e.to_string())?;
        let args = |d: &str, arret: bool| -> Vec<String> {
            let mut v: Vec<String> = [
                "gen",
                "--backend",
                b,
                "--torsion",
                "01,23",
                "--L",
                "8",
                "--beta",
                "2.5",
                "--graine",
                "26111902",
                "--therm",
                "2",
                "--nmeas",
                "6",
                "--nor",
                "4",
                "--reprise-every",
                "1",
                "--dump-every",
                "2",
                "--format",
                "q64",
                "--outdir",
                d,
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            if arret {
                v.extend(["--arret-apres".to_string(), "3".to_string()]);
            }
            v
        };
        let (dc, di, dl) = (
            base.join("complet"),
            base.join("interrompu"),
            base.join("ancien"),
        );
        let run = |a: Vec<String>| su2_cli(&a.iter().map(String::as_str).collect::<Vec<_>>());
        let rc = |o: &std::process::Output| o.status.code();
        let o = run(args(dc.to_str().unwrap(), false));
        if rc(&o) != Some(0) {
            return Err(format!(
                "run complet : {}",
                String::from_utf8_lossy(&o.stderr)
            ));
        }
        let o = run(args(di.to_str().unwrap(), true));
        if rc(&o) != Some(0) || di.join("COMPLETE").exists() {
            return Err(format!(
                "run interrompu : rc {:?} {}",
                rc(&o),
                String::from_utf8_lossy(&o.stderr)
            ));
        }
        let rj = std::fs::read_to_string(di.join("reprise.json")).map_err(|e| e.to_string())?;
        if !rj.contains(&format!("Geo64Gpu/{b}")) {
            return Err("reprise.json ne consigne pas le moteur C1".into());
        }
        let avant = fichiers(&di);
        let o = su2_cli(&[
            "gen",
            "--backend",
            "cpu",
            "--reprendre",
            di.to_str().unwrap(),
        ]);
        if rc(&o) != Some(2) || fichiers(&di) != avant {
            return Err(format!(
                "reprise CPU d'un run C1 : rc {:?} (2 attendu) ou fichiers modifiés",
                rc(&o)
            ));
        }
        // point de reprise sans invariants (binaire antérieur) : jamais repris par le chemin C1
        let o = std::process::Command::new("cp")
            .args(["-r", di.to_str().unwrap(), dl.to_str().unwrap()])
            .output();
        if !o.is_ok_and(|o| o.status.success()) {
            return Err("copie du run interrompu impossible".into());
        }
        let p = dl.join("reprise.json");
        let mut v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&p).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        v.as_object_mut()
            .ok_or("reprise.json n'est pas un objet")?
            .remove("invariants");
        std::fs::write(
            &p,
            serde_json::to_vec_pretty(&v).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let o = su2_cli(&["gen", "--backend", b, "--reprendre", dl.to_str().unwrap()]);
        if rc(&o) != Some(2) {
            return Err(format!(
                "point de reprise sans invariants repris par le chemin C1 : rc {:?}",
                rc(&o)
            ));
        }

        // c09c (point 1) : le même point de reprise ancien d'un run GPU, repris sur CPU : refusé (provenance GPU au manifeste)
        let avant_l = fichiers(&dl);
        let o = su2_cli(&[
            "gen",
            "--backend",
            "cpu",
            "--reprendre",
            dl.to_str().unwrap(),
        ]);
        if rc(&o) != Some(2) || fichiers(&dl) != avant_l {
            return Err(format!(
                "ancien point GPU tordu repris sur CPU : rc {:?} (2 attendu) ou fichiers modifiés",
                rc(&o)
            ));
        }
        // c09c (point 2) : invariants complets comparés AVANT l'ouverture du GPU : géométrie altérée (dir_temps 0) ⇒ 2, rien
        // d'écrit (manifeste compris)
        let dg = base.join("geometrie_alteree");
        let o = std::process::Command::new("cp")
            .args(["-r", di.to_str().unwrap(), dg.to_str().unwrap()])
            .output();
        if !o.is_ok_and(|o| o.status.success()) {
            return Err("copie du run interrompu impossible".into());
        }
        let pg = dg.join("reprise.json");
        let mut e: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&pg).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        e["invariants"]["geometrie"]["dir_temps"] = serde_json::json!(0);
        std::fs::write(
            &pg,
            serde_json::to_vec_pretty(&e).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let avant_g = fichiers(&dg);
        let o = su2_cli(&["gen", "--backend", b, "--reprendre", dg.to_str().unwrap()]);
        if rc(&o) != Some(2) || fichiers(&dg) != avant_g {
            return Err(format!(
                "géométrie consignée altérée : rc {:?} (2 attendu) ou fichiers modifiés",
                rc(&o)
            ));
        }
        let o = su2_cli(&["gen", "--backend", b, "--reprendre", di.to_str().unwrap()]);
        if rc(&o) != Some(0) || !di.join("COMPLETE").exists() {
            return Err(format!(
                "reprise --backend {b} : rc {:?} {}",
                rc(&o),
                String::from_utf8_lossy(&o.stderr)
            ));
        }
        for f in ["measurements.csv", "therm.csv"] {
            if std::fs::read(dc.join(f)).ok() != std::fs::read(di.join(f)).ok() {
                return Err(format!("{f} : run repris ≠ run complet"));
            }
        }
        // (b) invariants du moteur : gel, masque externe, moteur
        let cfg = lot(
            6,
            &[(2.5, 26111903, Depart::Chaud)],
            4,
            OptionsCompil::reference_k1(),
        );
        let spec = kvu::su2::geo_gpu::spec_su2_gen(6, Torsion::triviale());
        let neuf = |gel: bool| -> Result<Geo64Gpu, String> {
            let mut g = Geo64Gpu::nouveau(&app, cfg.clone(), &spec).map_err(|e| e.to_string())?;
            g.init().map_err(|e| e.to_string())?;
            if gel {
                g.fixer_gel(Gel::BordX0Spatial).map_err(|e| e.to_string())?;
            }
            Ok(g)
        };
        let dr = base.join("api");
        std::fs::create_dir_all(&dr).map_err(|e| e.to_string())?;
        let mut a = neuf(true)?;
        avancer(&mut a, 3, 4)?;
        kvu::su2::reprise::ecrire(&dr, &a, 3, 0).map_err(|e| e.to_string())?;
        let code_de =
            |r: Result<kvu::su2::reprise::EtatReprise, Erreur>| r.err().map(|e| e.code_sortie());
        let mut sans_gel = neuf(false)?;
        let mut autre_masque = neuf(true)?;
        autre_masque
            .fixer_masque(
                Some(&vec![0u8; 4 * 6usize.pow(4)]),
                kvu::su2::AppMasque::Tout,
            )
            .map_err(|e| e.to_string())?;
        let mut cpu = Cpu64::nouveau(cfg.clone()).map_err(|e| e.to_string())?;
        let refus = [
            (
                "sans gel",
                code_de(kvu::su2::reprise::lire(&dr, &mut sans_gel)),
            ),
            (
                "autre masque externe",
                code_de(kvu::su2::reprise::lire(&dr, &mut autre_masque)),
            ),
            (
                "moteur CPU",
                code_de(kvu::su2::reprise::lire(&dr, &mut cpu)),
            ),
        ];
        for (nom, c) in &refus {
            if *c != Some(code::USAGE) {
                return Err(format!(
                    "reprise moteur {nom} : code {c:?}, refus code 2 attendu"
                ));
            }
        }
        let mut repris = neuf(true)?;
        kvu::su2::reprise::lire(&dr, &mut repris).map_err(|e| e.to_string())?;
        avancer(&mut repris, 3, 4)?;
        let mut complet = neuf(true)?;
        avancer(&mut complet, 6, 4)?;
        let mots = comparer_champs(&repris, &complet, "gel restitué après reprise")?;
        let _ = std::fs::remove_dir_all(&base);
        Ok(format!(
            "backend {b} : reprise CPU d'un run C1 refusée (2, rien d'écrit), point de reprise sans invariants refusé au chemin \
             C1 (2), reprise {b} = run complet à l'octet (measurements.csv, therm.csv) ; moteur : sans gel / autre masque / CPU \
             refusés (2), gel restitué : {mots} mots égaux au run non interrompu"
        ))
    });
}
