//! Portes GPU du masque de liens gelés sur le chemin C1 (B10, lot C, côtés égaux, F64).
//! Toutes sont ignorées par défaut ; un SKIP ne vaut pas un résultat vert.
//!
//! Sous le verrou GPU de l'acemagic (`/tmp/kevinotron_gpu.lock`) :
//! ```text
//! cargo test -p kvu --release --test geo_gpu_masque -- --ignored --nocapture --test-threads=1
//! ```
//! Sur le gamer, sous `/tmp/gamer_gpu.lock` :
//! ```text
//! cargo test -p kvu --release --features cuda --test geo_gpu_masque -- --ignored --nocapture --test-threads=1
//! ```
//! Le backend suit KEVINOTRON_BACKEND, sinon CUDA avec la feature `cuda`, sinon HIP. Un GPU indisponible est un ÉCHEC
//! (relecture c09, O6) ; KVU_GPU_SAUT=1 demande explicitement de sauter (SKIP, qui ne vaut jamais vert).

use kvu::commun::{code, BackendKind, ChampHote, Depart, Erreur, Graine, Precision, Resultat};
use kvu::geom::{GeoSpec, SourceBord};
use kvu::gpu::{self, Appareil, OptionsCompil};
use kvu::su2::boite::{masque_boite, masque_gel_x0_spatial, GeometrieBoite};
use kvu::su2::cpu::Cpu64;
use kvu::su2::geo_gpu::Geo64Gpu;
use kvu::su2::ref64::Ref64Gpu;
use kvu::su2::torsion::Torsion;
use kvu::su2::{AppMasque, ConfigLot, Gel, NoyauxSu2, ParamChaine};

const CHAINES: [(f64, u64, Depart); 2] = [
    (2.5, 26111702, Depart::Chaud),
    (8.0, 26111703, Depart::Froid),
];

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

fn resynchroniser(gpu: &Geo64Gpu, cpu: &mut Cpu64) -> Result<(), String> {
    let q = gpu.telecharger(0).map_err(|e| e.to_string())?;
    cpu.televerser(0, &q).map_err(|e| e.to_string())?;
    cpu.fixer_sweep_id(gpu.sweep_id());
    Ok(())
}

fn masque_de_boite(ell: usize) -> Result<Vec<u8>, String> {
    let g = GeometrieBoite::nouvelle(ell).map_err(|e| e.to_string())?;
    Ok(masque_boite(&g))
}

/// Xorshift64* déterministe : environ un lien sur trois est gelé, sans dépendance supplémentaire.
fn masque_hasard(v: usize, graine: u64) -> Vec<u8> {
    let mut etat = if graine == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        graine
    };
    (0..4 * v)
        .map(|_| {
            etat ^= etat >> 12;
            etat ^= etat << 25;
            etat ^= etat >> 27;
            let tirage = etat.wrapping_mul(0x2545_F491_4F6C_DD1D);
            u8::from(tirage.is_multiple_of(3))
        })
        .collect()
}

/// Composantes des seuls liens gelés, dans l'ordre du champ ; une taille incohérente est un échec.
fn valeurs_gelees(masque: &[u8], q: &[f64]) -> Result<Vec<f64>, String> {
    if q.len() != 4 * masque.len() {
        return Err(format!(
            "champ de {} composantes pour un masque de {} liens : {} composantes attendues",
            q.len(),
            masque.len(),
            4 * masque.len()
        ));
    }
    Ok(masque
        .iter()
        .zip(q.chunks_exact(4))
        .filter(|(m, _)| **m != 0)
        .flat_map(|(_, lien)| lien.iter().copied())
        .collect())
}

fn comparer_geles(nom: &str, masque: &[u8], a: &[f64], b: &[f64]) -> Result<usize, String> {
    let ga = valeurs_gelees(masque, a)?;
    let gb = valeurs_gelees(masque, b)?;
    comparer_bits(nom, &ga, &gb)
}

fn liens_libres_modifies(masque: &[u8], a: &[f64], b: &[f64]) -> Result<(usize, usize), String> {
    if a.len() != 4 * masque.len() || b.len() != a.len() {
        return Err(format!(
            "liens libres : {} et {} composantes pour {} liens du masque",
            a.len(),
            b.len(),
            masque.len()
        ));
    }
    let (mut modifies, mut libres) = (0usize, 0usize);
    for ((&m, qa), qb) in masque.iter().zip(a.chunks_exact(4)).zip(b.chunks_exact(4)) {
        if m == 0 {
            libres += 1;
            modifies += usize::from(qa.iter().zip(qb).any(|(x, y)| x.to_bits() != y.to_bits()));
        }
    }
    if libres == 0 {
        return Err("0 lien libre : la fraction modifiée n'est pas définie".into());
    }
    Ok((modifies, libres))
}

fn cas_egal_c0(
    app: &Appareil,
    l: usize,
    masque: Option<&[u8]>,
    application: AppMasque,
    gel: Gel,
    contexte: &str,
) -> Result<usize, String> {
    let cfg = lot(l, &CHAINES, 4, OptionsCompil::reference_k1());
    let mut c0 = Ref64Gpu::nouveau(app, cfg.clone()).map_err(|e| e.to_string())?;
    let mut c1 = Geo64Gpu::nouveau(app, cfg, &GeoSpec::cubique(l, 0)).map_err(|e| e.to_string())?;
    c0.init().map_err(|e| e.to_string())?;
    c1.init().map_err(|e| e.to_string())?;
    c0.fixer_gel(gel).map_err(|e| e.to_string())?;
    c1.fixer_gel(gel).map_err(|e| e.to_string())?;
    c0.fixer_masque(masque, application)
        .map_err(|e| e.to_string())?;
    c1.fixer_masque(masque, application)
        .map_err(|e| e.to_string())?;
    let mut mots = 0;
    for pas in 1..=20 {
        if pas == 1 {
            let d0 = c0.sweep_hb_diag().map_err(|e| e.to_string())?;
            let d1 = c1.sweep_hb_diag().map_err(|e| e.to_string())?;
            comparer_exact(&format!("{contexte}, diagnostics du pas 1"), &d0, &d1)?;
        } else {
            c0.sweep_hb().map_err(|e| e.to_string())?;
            c1.sweep_hb().map_err(|e| e.to_string())?;
        }
        for _ in 0..4 {
            c0.sweep_or().map_err(|e| e.to_string())?;
            c1.sweep_or().map_err(|e| e.to_string())?;
        }
        if [1, 10, 20].contains(&pas) {
            let etape = format!("{contexte}, pas {pas}");
            mots += comparer_champs(&c0, &c1, &etape)?;
            mots += comparer_compteurs(&c0, &c1, &format!("{etape}, compteurs"))?;
        }
    }
    Ok(mots)
}

#[test]
#[ignore = "GPU"]
fn c_masque_egal_c0() {
    let Some(app) = appareil_ou_skip("C-MASQUE-C0") else {
        return;
    };
    gate("C-MASQUE-C0", || {
        let (mut cas, mut mots) = (0usize, 0usize);
        for (nom, l, masque) in [
            ("boîte ℓ = 4", 6, masque_de_boite(4)?),
            ("boîte ℓ = 6", 8, masque_de_boite(6)?),
            ("hyperplan", 6, masque_gel_x0_spatial(6)),
            ("hasard", 8, masque_hasard(8usize.pow(4), 26111701)),
        ] {
            for application in [
                AppMasque::Tout,
                AppMasque::IgnoreBain,
                AppMasque::IgnoreSurrelax,
                AppMasque::IgnoreTout,
            ] {
                let contexte = format!("L = {l}, {nom}, application {}", application.nom());
                mots += cas_egal_c0(&app, l, Some(&masque), application, Gel::Aucun, &contexte)?;
                cas += 1;
            }
        }
        let hasard = masque_hasard(6usize.pow(4), 26111704);
        for (nom, masque, application) in [
            ("gel seul", None, AppMasque::Tout),
            (
                "gel et hasard, bain sans masque externe",
                Some(hasard.as_slice()),
                AppMasque::IgnoreBain,
            ),
        ] {
            mots += cas_egal_c0(&app, 6, masque, application, Gel::BordX0Spatial, nom)?;
            cas += 1;
        }
        Ok(format!(
            "{cas} cas × 2 chaînes × 20 pas, diagnostics égaux au pas 1, \
             champs et compteurs aux pas 1/10/20 : {mots} mots de 64 bits égaux"
        ))
    });
}

#[test]
#[ignore = "GPU"]
fn c_masque_spec() {
    let Some(app) = appareil_ou_skip("C-MASQUE-SPEC") else {
        return;
    };
    gate("C-MASQUE-SPEC", || {
        let masque = masque_de_boite(4)?;
        let mut mots = 0;
        for (nom, spec, boite) in [
            (
                "boîte ℓ = 4",
                GeoSpec::boite(4, 0, SourceBord::ParChaine),
                true,
            ),
            (
                "hyperplan",
                GeoSpec::hyperplan(6, 0, SourceBord::ParChaine),
                false,
            ),
        ] {
            let cfg = lot(6, &CHAINES, 4, OptionsCompil::reference_k1());
            let mut geo = Geo64Gpu::nouveau(&app, cfg.clone(), &spec).map_err(|e| e.to_string())?;
            geo.init().map_err(|e| e.to_string())?;
            let mut manuel: Box<dyn NoyauxSu2> = if boite {
                let mut n = Geo64Gpu::nouveau(&app, cfg, &GeoSpec::cubique(6, 0))
                    .map_err(|e| e.to_string())?;
                n.init().map_err(|e| e.to_string())?;
                n.fixer_masque(Some(&masque), AppMasque::Tout)
                    .map_err(|e| e.to_string())?;
                Box::new(n)
            } else {
                let mut n = Ref64Gpu::nouveau(&app, cfg).map_err(|e| e.to_string())?;
                n.init().map_err(|e| e.to_string())?;
                n.fixer_gel(Gel::BordX0Spatial).map_err(|e| e.to_string())?;
                Box::new(n)
            };
            avancer(&mut geo, 10, 4)?;
            avancer(manuel.as_mut(), 10, 4)?;
            mots += comparer_champs(&geo, manuel.as_ref(), nom)?;
            mots += comparer_compteurs(&geo, manuel.as_ref(), &format!("{nom}, compteurs"))?;
        }
        Ok(format!(
            "2 géométries × 2 chaînes × 10 pas : {mots} mots de 64 bits égaux aux masques manuels C1/C0"
        ))
    });
}

#[test]
#[ignore = "GPU"]
fn c_masque_nul() {
    let Some(app) = appareil_ou_skip("C-MASQUE-NUL") else {
        return;
    };
    gate("C-MASQUE-NUL", || {
        let l = 6usize;
        let cfg = lot(l, &CHAINES, 4, OptionsCompil::reference_k1());
        let spec = GeoSpec::cubique(l, 0);
        let mut avec = Geo64Gpu::nouveau(&app, cfg.clone(), &spec).map_err(|e| e.to_string())?;
        let mut sans = Geo64Gpu::nouveau(&app, cfg, &spec).map_err(|e| e.to_string())?;
        avec.init().map_err(|e| e.to_string())?;
        sans.init().map_err(|e| e.to_string())?;
        let masque_nul = vec![0u8; 4 * l.pow(4)];
        avec.fixer_masque(Some(&masque_nul), AppMasque::Tout)
            .map_err(|e| e.to_string())?;
        avancer(&mut avec, 5, 4)?;
        avancer(&mut sans, 5, 4)?;
        let mut mots = comparer_champs(&avec, &sans, "masque nul contre absence de masque")?;
        mots += comparer_compteurs(&avec, &sans, "masque nul, compteurs")?;
        Ok(format!(
            "L = 6, 2 chaînes, 5 pas : {mots} mots de 64 bits égaux"
        ))
    });
}

#[derive(Default)]
struct CasCpuGpu {
    divergentes: usize,
    pire: f64,
    mots_geles: usize,
    divergentes_libres: usize,
    ecart_libre: f64,
}

/// Les deux comparaisons au début du pas imposent aussi l'égalité des liens gelés entre CPU et GPU.
fn comparer_balayage(
    gpu: &Geo64Gpu,
    cpu: &Cpu64,
    masque: &[u8],
    initial: &ChampHote,
    contexte: &str,
) -> Result<(f64, usize), String> {
    let qg = gpu.telecharger(0).map_err(|e| e.to_string())?;
    let qc = cpu.telecharger(0).map_err(|e| e.to_string())?;
    let mut mots = comparer_geles(
        &format!("{contexte}, liens gelés GPU contre début du pas"),
        masque,
        &qg.q,
        &initial.q,
    )?;
    mots += comparer_geles(
        &format!("{contexte}, liens gelés CPU contre début du pas"),
        masque,
        &qc.q,
        &initial.q,
    )?;
    Ok((ecart_max(&qg.q, &qc.q), mots))
}

/// Procédure de cas_tordu, sans torsion : 10 pas resynchronisés, puis un pas libre depuis un état commun.
fn cas_cpu_gpu(app: &Appareil, l: usize, masque: &[u8], beta: f64) -> Result<CasCpuGpu, String> {
    let cfg = lot(
        l,
        &[(beta, 26111702, Depart::Chaud)],
        4,
        OptionsCompil::reference_k1(),
    );
    let mut gpu =
        Geo64Gpu::nouveau(app, cfg.clone(), &GeoSpec::cubique(l, 0)).map_err(|e| e.to_string())?;
    gpu.init().map_err(|e| e.to_string())?;
    let mut cpu = Cpu64::nouveau(cfg).map_err(|e| e.to_string())?;
    cpu.init().map_err(|e| e.to_string())?;
    gpu.fixer_masque(Some(masque), AppMasque::Tout)
        .map_err(|e| e.to_string())?;
    cpu.fixer_masque(Some(masque), AppMasque::Tout)
        .map_err(|e| e.to_string())?;
    let mut cas = CasCpuGpu::default();
    for pas in 1..=10 {
        resynchroniser(&gpu, &mut cpu)?;
        let initial = gpu.telecharger(0).map_err(|e| e.to_string())?;
        let dg = gpu.sweep_hb_diag().map_err(|e| e.to_string())?;
        let dc = cpu.sweep_hb_diag().map_err(|e| e.to_string())?;
        cas.divergentes += compter_ecarts(&dg, &dc);
        let contexte = format!("L = {l}, β = {beta}, pas {pas}, bain");
        let (ecart, mots) = comparer_balayage(&gpu, &cpu, masque, &initial, &contexte)?;
        cas.pire = cas.pire.max(ecart);
        cas.mots_geles += mots;
        for surrelax in 1..=4 {
            resynchroniser(&gpu, &mut cpu)?;
            gpu.sweep_or().map_err(|e| e.to_string())?;
            cpu.sweep_or().map_err(|e| e.to_string())?;
            let contexte = format!("L = {l}, β = {beta}, pas {pas}, surrelaxation {surrelax}");
            let (ecart, mots) = comparer_balayage(&gpu, &cpu, masque, &initial, &contexte)?;
            cas.pire = cas.pire.max(ecart);
            cas.mots_geles += mots;
        }
    }
    // Une seule resynchronisation, avant les cinq balayages du pas libre.
    resynchroniser(&gpu, &mut cpu)?;
    let initial = gpu.telecharger(0).map_err(|e| e.to_string())?;
    let dg = gpu.sweep_hb_diag().map_err(|e| e.to_string())?;
    let dc = cpu.sweep_hb_diag().map_err(|e| e.to_string())?;
    cas.divergentes_libres = compter_ecarts(&dg, &dc);
    for _ in 0..4 {
        gpu.sweep_or().map_err(|e| e.to_string())?;
        cpu.sweep_or().map_err(|e| e.to_string())?;
    }
    let contexte = format!("L = {l}, β = {beta}, pas libre");
    let (ecart, mots) = comparer_balayage(&gpu, &cpu, masque, &initial, &contexte)?;
    cas.ecart_libre = ecart;
    cas.mots_geles += mots;
    Ok(cas)
}

#[test]
#[ignore = "GPU"]
fn c_masque_cpu_gpu() {
    let Some(app) = appareil_ou_skip("C-MASQUE-CPU-GPU") else {
        verdict("SKIP C-MASQUE-CPU-GPU-MUTANT : 0 cas exécuté, même appareil indisponible");
        return;
    };
    gate("C-MASQUE-CPU-GPU", || {
        let (mut cas, mut divergentes, mut mots_geles, mut divergentes_libres) =
            (0usize, 0usize, 0usize, 0usize);
        let (mut pire, mut ecart_libre) = (0.0f64, 0.0f64);
        for (l, masque) in [
            (6, masque_de_boite(4)?),
            (8, masque_de_boite(6)?),
            (8, masque_hasard(8usize.pow(4), 26111705)),
        ] {
            for beta in [2.5, 8.0] {
                let r = cas_cpu_gpu(&app, l, &masque, beta)?;
                divergentes += r.divergentes;
                pire = pire.max(r.pire);
                mots_geles += r.mots_geles;
                divergentes_libres += r.divergentes_libres;
                ecart_libre = ecart_libre.max(r.ecart_libre);
                cas += 1;
            }
        }
        let detail = format!(
            "{cas} cas × 10 pas, 50 balayages par cas : {divergentes} décisions divergentes, max|Δq| = {pire:.3e} \
             (≤ 1e-12) ; {cas} pas libres : {divergentes_libres} décisions divergentes, max|Δq| = {ecart_libre:.3e} \
             (≤ 1e-10) ; {mots_geles} mots gelés identiques à l'état initial du pas"
        );
        if cas == 6
            && divergentes == 0
            && pire <= 1e-12
            && divergentes_libres == 0
            && ecart_libre <= 1e-10
        {
            Ok(detail)
        } else {
            Err(detail)
        }
    });
    gate("C-MASQUE-CPU-GPU-MUTANT", || {
        let l = 6;
        let masque = masque_de_boite(4)?;
        let cfg = lot(
            l,
            &[(2.5, 26111702, Depart::Chaud)],
            4,
            OptionsCompil::reference_k1(),
        );
        let mut gpu = Geo64Gpu::nouveau(&app, cfg.clone(), &GeoSpec::cubique(l, 0))
            .map_err(|e| e.to_string())?;
        gpu.init().map_err(|e| e.to_string())?;
        let mut cpu = Cpu64::nouveau(cfg).map_err(|e| e.to_string())?;
        cpu.init().map_err(|e| e.to_string())?;
        gpu.fixer_masque(Some(&masque), AppMasque::Tout)
            .map_err(|e| e.to_string())?;
        cpu.fixer_masque(None, AppMasque::Tout)
            .map_err(|e| e.to_string())?;
        resynchroniser(&gpu, &mut cpu)?;
        let initial = gpu.telecharger(0).map_err(|e| e.to_string())?;
        let dg = gpu.sweep_hb_diag().map_err(|e| e.to_string())?;
        let dc = cpu.sweep_hb_diag().map_err(|e| e.to_string())?;
        let divergentes = compter_ecarts(&dg, &dc);
        for _ in 0..4 {
            gpu.sweep_or().map_err(|e| e.to_string())?;
            cpu.sweep_or().map_err(|e| e.to_string())?;
        }
        let qg = gpu.telecharger(0).map_err(|e| e.to_string())?;
        let qc = cpu.telecharger(0).map_err(|e| e.to_string())?;
        comparer_geles(
            "mutant, liens gelés GPU intacts",
            &masque,
            &qg.q,
            &initial.q,
        )?;
        let gg = valeurs_gelees(&masque, &qg.q)?;
        let gc = valeurs_gelees(&masque, &qc.q)?;
        let differences = compter_ecarts(&bits(&gg), &bits(&gc));
        let detail = format!(
            "1 cas, L = 6, β = 2,5, 1 pas : CPU sans masque, {divergentes} décisions divergentes, \
             {differences} mots gelés différents ; mutant attendu ROUGE"
        );
        if divergentes > 0 || differences > 0 {
            Ok(detail)
        } else {
            Err(detail)
        }
    });
}

#[test]
#[ignore = "GPU"]
fn c_masque_geles_intacts() {
    let Some(app) = appareil_ou_skip("C-MASQUE-GELES-INTACTS") else {
        verdict("SKIP C-MASQUE-GELES-MUTANTS : 0 cas exécuté, même appareil indisponible");
        return;
    };
    gate("C-MASQUE-GELES-INTACTS", || {
        let (mut cas, mut mots) = (0usize, 0usize);
        let mut fractions = Vec::new();
        for l in [6, 8] {
            let masque = masque_de_boite(l - 2)?;
            let cfg = lot(
                l,
                &[
                    (2.5, 26111706, Depart::Chaud),
                    (2.5, 26111707, Depart::Froid),
                ],
                4,
                OptionsCompil::reference_k1(),
            );
            let mut gpu =
                Geo64Gpu::nouveau(&app, cfg, &GeoSpec::cubique(l, 0)).map_err(|e| e.to_string())?;
            gpu.init().map_err(|e| e.to_string())?;
            gpu.fixer_masque(Some(&masque), AppMasque::Tout)
                .map_err(|e| e.to_string())?;
            let avant: Vec<ChampHote> = (0..2)
                .map(|c| gpu.telecharger(c))
                .collect::<Resultat<_>>()
                .map_err(|e| e.to_string())?;
            for pas in 1..=50 {
                avancer(&mut gpu, 1, 4)?;
                for (c, initial) in avant.iter().enumerate() {
                    let final_ = gpu.telecharger(c).map_err(|e| e.to_string())?;
                    let contexte = format!("L = {l}, chaîne {c}, pas {pas}");
                    mots += comparer_geles(&contexte, &masque, &initial.q, &final_.q)?;
                    if pas == 5 {
                        let (modifies, libres) =
                            liens_libres_modifies(&masque, &initial.q, &final_.q)?;
                        let fraction = modifies as f64 / libres as f64;
                        let depart = if c == 0 { "chaude" } else { "froide" };
                        let detail = format!(
                            "L = {l}, chaîne {depart} : {modifies}/{libres} = {fraction:.6}"
                        );
                        if fraction < 0.99 {
                            return Err(format!("{detail} < 0,99 après 5 pas"));
                        }
                        fractions.push(detail);
                    }
                }
            }
            cas += avant.len();
        }
        Ok(format!(
            "{cas} cas × 50 pas, {mots} mots gelés égaux à l'état initial ; \
             fractions de liens libres modifiés après 5 pas : {}",
            fractions.join(" ; ")
        ))
    });
    gate("C-MASQUE-GELES-MUTANTS", || {
        let masque = masque_de_boite(4)?;
        let mut details = Vec::new();
        for application in [
            AppMasque::IgnoreTout,
            AppMasque::IgnoreBain,
            AppMasque::IgnoreSurrelax,
        ] {
            let cfg = lot(
                6,
                &[(2.5, 26111706, Depart::Chaud)],
                4,
                OptionsCompil::reference_k1(),
            );
            let mut gpu =
                Geo64Gpu::nouveau(&app, cfg, &GeoSpec::cubique(6, 0)).map_err(|e| e.to_string())?;
            gpu.init().map_err(|e| e.to_string())?;
            gpu.fixer_masque(Some(&masque), application)
                .map_err(|e| e.to_string())?;
            let initial = gpu.telecharger(0).map_err(|e| e.to_string())?;
            avancer(&mut gpu, 5, 4)?;
            let final_ = gpu.telecharger(0).map_err(|e| e.to_string())?;
            let a = valeurs_gelees(&masque, &initial.q)?;
            let b = valeurs_gelees(&masque, &final_.q)?;
            let ecart = ecart_max(&a, &b);
            let detail = format!("{} : max|Δq| gelé = {ecart:.3e}", application.nom());
            if !ecart.is_finite() || ecart <= 0.1 {
                return Err(format!("{detail}, seuil > 0,1 non satisfait"));
            }
            details.push(detail);
        }
        Ok(format!(
            "3/3 mutants ROUGES après 5 pas, L = 6 : {}",
            details.join(" ; ")
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
fn refus_lot_c() {
    let Some(app) = appareil_ou_skip("REFUS-LOT-C") else {
        return;
    };
    gate("REFUS-LOT-C", || {
        let cfg = lot(
            4,
            &[(2.5, 26111702, Depart::Chaud)],
            4,
            OptionsCompil::reference_k1(),
        );
        let torsion = Torsion::analyser("01").map_err(|e| e.to_string())?;
        let spec = GeoSpec::cubique(4, 0).avec_torsion(torsion);
        let mut tordu = Geo64Gpu::nouveau(&app, cfg.clone(), &spec).map_err(|e| e.to_string())?;
        let masque_nul = vec![0u8; 4 * cfg.l.pow(4)];
        verifier_refus(
            "torsion et masque nul",
            tordu.fixer_masque(Some(&masque_nul), AppMasque::Tout),
            code::USAGE,
        )?;
        verifier_refus(
            "torsion et gel",
            tordu.fixer_gel(Gel::BordX0Spatial),
            code::USAGE,
        )?;
        let mut trivial = Geo64Gpu::nouveau(&app, cfg.clone(), &GeoSpec::cubique(4, 0))
            .map_err(|e| e.to_string())?;
        verifier_refus(
            "masque de 10 octets au lieu de 1024",
            trivial.fixer_masque(Some(&[0u8; 10]), AppMasque::Tout),
            code::USAGE,
        )?;
        // O4 : valeurs de bord prescrites que le moteur ne pose pas (accepté en silence sur 08d4c84).
        let cfg6 = lot(6, &CHAINES, 4, OptionsCompil::reference_k1());
        verifier_refus(
            "hyperplan froid (valeurs prescrites)",
            Geo64Gpu::nouveau(
                &app,
                cfg6.clone(),
                &GeoSpec::hyperplan(6, 0, SourceBord::Froid),
            ),
            code::USAGE,
        )?;
        verifier_refus(
            "boîte froide (valeurs prescrites)",
            Geo64Gpu::nouveau(&app, cfg6, &GeoSpec::boite(4, 0, SourceBord::Froid)),
            code::USAGE,
        )?;
        verifier_refus(
            "boîte ℓ = 4, côtés 6 contre cfg.l = 4",
            Geo64Gpu::nouveau(&app, cfg, &GeoSpec::boite(4, 0, SourceBord::Froid)),
            code::USAGE,
        )?;
        Ok(
            "6/6 refus : torsion et masque nul, torsion et gel, taille 10, hyperplan et boîte aux valeurs prescrites ⇒ Config \
            (code 2) ; côtés 6 contre cfg.l = 4 ⇒ Config (code 2), variantes et codes vérifiés"
                .into(),
        )
    });
}
