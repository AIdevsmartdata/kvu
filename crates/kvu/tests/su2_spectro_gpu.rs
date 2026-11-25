//! Tests GPU de `su2 spectro gpu` (`#[ignore = "GPU"]`), à lancer sous le verrou :
//! `KVU_F32_SOUS_VERROU=1 flock -w 3600 /tmp/gamer_gpu.lock cargo test --release --features cuda --test su2_spectro_gpu -- --ignored --nocapture --test-threads=1`
//! S1 champ froid (réponse connue) ; S2 égalité avec le processeur (écart relatif maximal rapporté) ; S3 invariance de jauge ;
//! S4 translation ; S5 invariance cubique (permutations d'axes ET réflexions : k ↔ −k) ; S6 contrôles négatifs (mutants).
use kvu::commun::{BackendKind, Precision};
use kvu::gpu::Appareil;
use kvu::heritage::su2_hb::{qdag, Q};
use kvu::su2::spectro::gpu::{MesCh, ParamGpu, SpectroGpu};
use kvu::su2::spectro::neutre::{self, MutantN, ParamNeutre, FORMES};
use kvu::su2::spectro::transformer_jauge;
use kvu::su2::tube::reseau::GeoA;
use kvu::su2::spectro::bloc::{self, ParamBloc};

const COTES: [usize; 4] = [6, 6, 6, 8];
const NCH: usize = 3;

fn appareil() -> Option<Appareil> {
    if std::env::var("KVU_F32_SOUS_VERROU").as_deref() != Ok("1") {
        println!("SKIP (pas sous verrou GPU)");
        return None;
    }
    let b = if cfg!(feature = "cuda") { BackendKind::Cuda } else { BackendKind::Hip };
    kvu::gpu::ouvrir(b, 0).ok()
}

fn param(prec: Precision, mutant: u8) -> ParamGpu {
    ParamGpu {
        cotes: COTES,
        temps: 3,
        niveaux: vec![0, 2, 4],
        alpha: 0.5,
        formes: FORMES.iter().map(|f| f.0.to_string()).collect(),
        tau_max: 3,
        ks: vec![1, 2],
        precision: prec,
        mutant,
        nor: 4,
        reps: vec![],
        betas: vec![2.3; NCH],
        graines: vec![11, 12, 13],
        bloc: ParamBloc::vide(), ops_bruts: false, xi0: None, orbites: vec![],
    }
}

fn en_q(raw: &[f64]) -> Vec<Q> {
    raw.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
}
fn en_raw(u: &[Q]) -> Vec<f64> {
    u.iter().flatten().copied().collect()
}

fn cpu(g: &GeoA, u: &[Q], p: &ParamGpu) -> neutre::MesureNeutre {
    let formes: Vec<usize> = p.formes.iter().map(|n| FORMES.iter().position(|f| f.0 == n).unwrap()).collect();
    let pn = ParamNeutre { temps: p.temps, niveaux: p.niveaux.clone(), alpha: p.alpha, formes, tau_max: p.tau_max, mutant: MutantN::Aucun };
    if p.bloc.actif() {
        bloc::mesurer_avec_blocs(g, u, &pn, &p.ks, &p.bloc).unwrap()
    } else {
        neutre::mesurer_ps(g, u, &pn, &p.ks).unwrap()
    }
}

/// (écart relatif maximal sur les entrées > 1e-6·max, écart normalisé au maximum du tableau)
fn ecart(a: &[f64], b: &[f64]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let sc = b.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let (mut r, mut n) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        assert!(x.is_finite() && y.is_finite());
        if y.abs() > 1e-6 * sc {
            r = r.max((x - y).abs() / y.abs());
        }
        n = n.max((x - y).abs() / sc.max(1e-300));
    }
    (r, n)
}

fn tous(m: &MesCh) -> Vec<f64> {
    let mut v = m.corr.clone();
    for p in &m.corr_ps {
        v.extend_from_slice(p);
    }
    v.extend_from_slice(&m.moy);
    v
}

/// Mesure du champ `u` posé dans la chaîne 0 (les autres chaînes gardent leur champ).
fn mesure_champ(sg: &mut SpectroGpu, u: &[Q]) -> MesCh {
    sg.boite.televerser(0, &en_raw(u)).unwrap();
    sg.mesurer().unwrap().remove(0)
}

fn translation(g: &GeoA, u: &[Q], dx: [usize; 4]) -> Vec<Q> {
    let mut out = u.to_vec();
    for mu in 0..4 {
        for s in 0..g.v {
            let x = g.coords(s);
            let mut y = [0usize; 4];
            for d in 0..4 {
                y[d] = (x[d] + g.d[d] - dx[d] % g.d[d]) % g.d[d];
            }
            out[mu * g.v + s] = u[mu * g.v + g.site(&y)];
        }
    }
    out
}

/// Rotation/réflexion cubique des trois axes spatiaux (0,1,2) : x'_i = ± x_{π(i)} ; U'_i(x') = U_{π(i)}(x) ou U_{π(i)}(x − e)† si l'axe est réfléchi.
fn cubique(g: &GeoA, u: &[Q], perm: [usize; 3], flip: [bool; 3]) -> Vec<Q> {
    let mut out = u.to_vec();
    for s in 0..g.v {
        let xn = g.coords(s);
        let mut x = xn;
        for i in 0..3 {
            let l = g.d[perm[i]];
            x[perm[i]] = if flip[i] { (l - xn[i]) % l } else { xn[i] };
        }
        for i in 0..3 {
            let j = perm[i];
            out[i * g.v + s] = if !flip[i] {
                u[j * g.v + g.site(&x)]
            } else {
                let mut y = x;
                y[j] = (x[j] + g.d[j] - 1) % g.d[j];
                qdag(u[j * g.v + g.site(&y)])
            };
        }
    }
    out
}

struct Bilan {
    froid: f64,
    cpu_rel: f64,
    cpu_norm: f64,
    jauge: f64,
    transl: f64,
    cub: f64,
}

fn bilan(app: &Appareil, prec: Precision, mutant: u8) -> Bilan {
    bilan_p(app, &param(prec, mutant))
}

fn bilan_p(app: &Appareil, p: &ParamGpu) -> Bilan {
    let p = p.clone();
    let g = GeoA::analyser("6,6,6,8").unwrap();
    let mut sg = SpectroGpu::nouveau(app, &p).unwrap();
    sg.init().unwrap();
    sg.avancer(40).unwrap();
    // configurations différentes par chaîne
    let cfgs: Vec<Vec<Q>> = (0..NCH).map(|c| en_q(&sg.boite.telecharger(c).unwrap())).collect();
    // S2 : toutes les chaînes d'un coup contre le processeur
    let res = sg.mesurer().unwrap();
    let (mut cr, mut cn) = (0.0f64, 0.0f64);
    for c in 0..NCH {
        let r = cpu(&g, &cfgs[c], &p);
        let mut ref_t: Vec<f64> = r.corr.iter().flatten().copied().collect();
        for q in &r.corr_ps {
            ref_t.extend(q.iter().flatten());
        }
        ref_t.extend(&r.moy);
        let (a, b) = ecart(&tous(&res[c]), &ref_t);
        cr = cr.max(a);
        cn = cn.max(b);
    }
    // S1 : champ froid : S = 1 (donc C = 1, moy = 1), impulsions nulles
    let froid_u = vec![[1.0, 0.0, 0.0, 0.0]; g.v * 4];
    let m = mesure_champ(&mut sg, &froid_u);
    let mut froid = 0.0f64;
    for x in &m.corr {
        froid = froid.max((x - 1.0).abs());
    }
    for x in &m.moy {
        froid = froid.max((x - 1.0).abs());
    }
    for p in &m.corr_ps {
        for x in p {
            froid = froid.max(x.abs());
        }
    }
    let base_u = &cfgs[0];
    let ref_m = mesure_champ(&mut sg, base_u);
    let rv = tous(&ref_m);
    // S3 : jauge
    let mut uj = base_u.clone();
    transformer_jauge(&g, &mut uj, 777);
    let jauge = ecart(&tous(&mesure_champ(&mut sg, &uj)), &rv).1;
    // S4 : translation
    let transl = ecart(&tous(&mesure_champ(&mut sg, &translation(&g, base_u, [1, 2, 3, 5]))), &rv).1;
    // S5 : cubique : les 6 permutations × 8 réflexions (48 éléments), plus de 24 rotations
    let perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    let mut cub = 0.0f64;
    for pm in perms {
        for k in 0..8u8 {
            let fl = [k & 1 != 0, k & 2 != 0, k & 4 != 0];
            let uc = cubique(&g, base_u, pm, fl);
            cub = cub.max(ecart(&tous(&mesure_champ(&mut sg, &uc)), &rv).1);
        }
    }
    Bilan { froid, cpu_rel: cr, cpu_norm: cn, jauge, transl, cub }
}

fn tolerance(prec: Precision) -> f64 {
    if prec == Precision::F64 {
        1e-12
    } else {
        1e-5
    }
}

#[test]
#[ignore = "GPU"]
fn spectro_gpu_validation_f64_et_f32() {
    let Some(app) = appareil() else { return };
    for prec in [Precision::F64, Precision::F32] {
        let b = bilan(&app, prec, 0);
        let tol = tolerance(prec);
        println!(
            "BILAN {} : froid {:.2e} ; cpu rel max {:.3e} (norm {:.3e}) ; jauge {:.2e} ; translation {:.2e} ; cubique(48) {:.2e}",
            prec.nom(), b.froid, b.cpu_rel, b.cpu_norm, b.jauge, b.transl, b.cub
        );
        assert!(b.froid < tol * 10.0, "champ froid : {}", b.froid);
        assert!(b.cpu_norm < tol * 10.0, "contre le processeur : {}", b.cpu_norm);
        assert!(b.jauge < tol * 10.0, "jauge : {}", b.jauge);
        assert!(b.transl < tol * 10.0, "translation : {}", b.transl);
        assert!(b.cub < tol * 10.0, "cubique : {}", b.cub);
    }
}

/// Chaque mutant doit faire échouer au moins une des portes (F64, tolérance 1e-9).
#[test]
#[ignore = "GPU"]
fn spectro_gpu_mutants_tous_detectes() {
    let Some(app) = appareil() else { return };
    for mutant in 1..=5u8 {
        let b = bilan(&app, Precision::F64, mutant);
        let echec = b.froid > 1e-9 || b.cpu_norm > 1e-9 || b.jauge > 1e-9 || b.transl > 1e-9 || b.cub > 1e-9;
        println!(
            "MUTANT {mutant} : froid {:.1e} cpu {:.1e} jauge {:.1e} transl {:.1e} cubique {:.1e} -> {}",
            b.froid, b.cpu_norm, b.jauge, b.transl, b.cub, if echec { "DÉTECTÉ" } else { "NON DÉTECTÉ" }
        );
        assert!(echec, "mutant {mutant} non détecté");
    }
}

// ---------------------------------------------------------------------------------------------------------------------------
// Représentations cubiques A2, E, T1, T2 (± parité) : GPU contre une implémentation hôte indépendante (formes → images → poids)
// ---------------------------------------------------------------------------------------------------------------------------
use kvu::su2::spectro::cubique::{analyser_rep, poids_images};
use kvu::su2::spectro::ape_iteration;

/// Corrélateurs de la représentation `nom` (moyenne des lignes) : (corr aplati [paire][τ], moy), formes `retenues`.
fn cpu_rep(g: &GeoA, u: &[Q], p: &ParamGpu, nom: &str, retenues: &[String]) -> (Vec<f64>, Vec<f64>) {
    let (ir, par) = analyser_rep(nom).unwrap();
    let mut champs = vec![u.to_vec()];
    for k in 1..=*p.niveaux.last().unwrap() {
        champs.push(ape_iteration(g, &champs[k - 1], p.temps, p.alpha));
    }
    let mut corr_tot: Vec<f64> = Vec::new();
    let mut moy_tot: Vec<f64> = Vec::new();
    for row in 0..ir.dim() * ir.dim() {
        let mut s: Vec<Vec<f64>> = Vec::new();
        for &n in &p.niveaux {
            for nm in retenues {
                let (_, ch) = *FORMES.iter().find(|f| f.0 == nm).unwrap();
                let w = poids_images(ch, ir, par, row / ir.dim(), row % ir.dim());
                let imgs = neutre::images(ch);
                let mut ser = vec![0.0; g.d[p.temps]];
                for (k, im) in imgs.iter().enumerate() {
                    let t = neutre::tranches(g, &champs[n], p.temps, std::slice::from_ref(im));
                    for (a, b) in ser.iter_mut().zip(t) {
                        *a += w[k] * b;
                    }
                }
                s.push(ser);
            }
        }
        let c: Vec<f64> = neutre::correlateurs(&s, p.tau_max).into_iter().flatten().collect();
        let m: Vec<f64> = s.iter().map(|x| x.iter().sum::<f64>() / x.len() as f64).collect();
        if corr_tot.is_empty() {
            corr_tot = vec![0.0; c.len()];
            moy_tot = vec![0.0; m.len()];
        }
        for (a, b) in corr_tot.iter_mut().zip(&c) {
            *a += b / (ir.dim() * ir.dim()) as f64;
        }
        for (a, b) in moy_tot.iter_mut().zip(&m) {
            *a += b / (ir.dim() * ir.dim()) as f64;
        }
    }
    (corr_tot, moy_tot)
}

#[test]
#[ignore = "GPU"]
fn spectro_gpu_representations_cubiques() {
    let Some(app) = appareil() else { return };
    let mut p = param(Precision::F64, 0);
    p.reps = ["A2+", "E+", "T1+", "T2+", "A1-", "A2-", "E-", "T1-", "T2-"].iter().map(|s| s.to_string()).collect();
    let g = GeoA::analyser("6,6,6,8").unwrap();
    let mut sg = SpectroGpu::nouveau(&app, &p).unwrap();
    sg.init().unwrap();
    sg.avancer(40).unwrap();
    let cfgs: Vec<Vec<Q>> = (0..NCH).map(|c| en_q(&sg.boite.telecharger(c).unwrap())).collect();
    let res = sg.mesurer().unwrap();
    let mut pire = 0.0f64;
    for (i, rp) in sg.reps.iter().enumerate() {
        println!("REP {} : {} formes non nulles : {:?}", rp.nom, rp.formes.len(), rp.formes);
        for c in 0..NCH {
            let (cc, mm) = cpu_rep(&g, &cfgs[c], &p, &rp.nom, &rp.formes);
            let mut a = res[c].reps[i].0.clone();
            a.extend_from_slice(&res[c].reps[i].1);
            let mut b = cc;
            b.extend_from_slice(&mm);
            let (_, n) = ecart(&a, &b);
            pire = pire.max(n);
        }
    }
    println!("REPS contre hôte : écart normalisé max {pire:.3e}");
    assert!(pire < 1e-11);
    // champ froid : toutes les représentations non triviales s'annulent
    let froid_u = vec![[1.0, 0.0, 0.0, 0.0]; g.v * 4];
    let m = mesure_champ(&mut sg, &froid_u);
    let fr = m.reps.iter().flat_map(|(c, mo)| c.iter().chain(mo.iter())).fold(0.0f64, |a, x| a.max(x.abs()));
    println!("REPS champ froid : max |.| = {fr:.2e}");
    assert!(fr < 1e-13);
    // invariance sous les 48 éléments (les corrélateurs sommés sur les lignes sont invariants), et sous la jauge
    let base_u = &cfgs[0];
    let ref_m = mesure_champ(&mut sg, base_u);
    let rv: Vec<f64> = ref_m.reps.iter().flat_map(|(c, _)| c.iter().copied()).collect();
    let perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    let mut cub = 0.0f64;
    for pm in perms {
        for k in 0..8u8 {
            let uc = cubique(&g, base_u, pm, [k & 1 != 0, k & 2 != 0, k & 4 != 0]);
            let v: Vec<f64> = mesure_champ(&mut sg, &uc).reps.iter().flat_map(|(c, _)| c.iter().copied()).collect();
            cub = cub.max(ecart(&v, &rv).1);
        }
    }
    let mut uj = base_u.clone();
    transformer_jauge(&g, &mut uj, 4242);
    let vj: Vec<f64> = mesure_champ(&mut sg, &uj).reps.iter().flat_map(|(c, _)| c.iter().copied()).collect();
    let jg = ecart(&vj, &rv).1;
    println!("REPS invariance cubique(48) {cub:.2e} ; jauge {jg:.2e}");
    assert!(cub < 1e-12 && jg < 1e-12);
}

// ---------------------------------------------------------------------------------------------------------------------------
// Torelons sur carte contre `spectro::mesurer` (CPU F64)
// ---------------------------------------------------------------------------------------------------------------------------
use kvu::su2::spectro::gpu_torelon::{ParamTorelon, TorelonGpu};
use kvu::su2::spectro::{mesurer, Axes, Mutant, ParamMesure};

#[test]
#[ignore = "GPU"]
#[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
fn spectro_gpu_torelon_contre_cpu() {
    let Some(app) = appareil() else { return };
    let g = GeoA::analyser("6,6,6,8").unwrap();
    for prec in [Precision::F64, Precision::F32] {
        let p = ParamTorelon {
            cotes: COTES, w: 0, temps: 3, niveaux: vec![0, 2, 4, 8], alpha: 0.5, moments: vec![[0, 0], [1, 0], [1, 1]], tau_max: 4,
            precision: prec, nor: 4, betas: vec![2.3; NCH], graines: vec![21, 22, 23], bloc: ParamBloc::vide(), xi0: None, ops_bruts: false, profil: vec![], profil_niveau: None,
        };
        let mut tg = TorelonGpu::nouveau(&app, &p).unwrap();
        tg.init().unwrap();
        tg.avancer(40).unwrap();
        let res = tg.mesurer().unwrap();
        let pm = ParamMesure { axes: Axes::nouveaux(0, 3).unwrap(), niveaux: p.niveaux.clone(), alpha: 0.5, moments: p.moments.clone(), tau_max: 4, mutant: Mutant::Aucun };
        let (mut pire, mut pire_rel) = (0.0f64, 0.0f64);
        for c in 0..NCH {
            let u = en_q(&tg.boite.telecharger(c).unwrap());
            let r = mesurer(&g, &u, &pm).unwrap();
            for i in 0..p.moments.len() {
                let a: Vec<f64> = res[c].corr[i].iter().flatten().copied().collect();
                let b: Vec<f64> = r.corr[i].iter().flatten().copied().collect();
                let (rr, n) = ecart(&a, &b);
                pire = pire.max(n);
                pire_rel = pire_rel.max(rr);
                let a: Vec<f64> = res[c].phi_moyen[i].iter().flatten().copied().collect();
                let b: Vec<f64> = r.phi_moyen[i].iter().flatten().copied().collect();
                pire = pire.max(ecart(&a, &b).1);
            }
        }
        println!("TORELON {} contre CPU : écart normalisé max {pire:.3e}, relatif max {pire_rel:.3e}", prec.nom());
        assert!(pire < tolerance(prec) * 10.0);
        // champ froid : P = 1 partout ⇒ C(p = 0) = 1, C(p ≠ 0) = 0, Polyakov = 1, plaquette = 1
        tg.boite.televerser(0, &en_raw(&vec![[1.0, 0.0, 0.0, 0.0]; g.v * 4])).unwrap();
        let f = tg.mesurer().unwrap().remove(0);
        let e0 = f.corr[0].iter().flatten().fold(0.0f64, |m, x| m.max((x - 1.0).abs()));
        let e1 = f.corr[1].iter().chain(f.corr[2].iter()).flatten().fold(0.0f64, |m, x| m.max(x.abs()));
        println!("TORELON {} champ froid : {e0:.2e} {e1:.2e} poly {} plaq {}", prec.nom(), f.poly_w, f.plaq);
        assert!(e0 < 1e-13 && e1 < 1e-13 && (f.poly_w - 1.0).abs() < 1e-14 && (f.plaq - 1.0).abs() < 1e-14);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------
// BLOCAGE DE TEPER (branche spectro/blocage-1120) : carte contre `su2::spectro::bloc` (processeur F64), réponses connues,
// invariances, mutants 6 à 8, identité au bit des anciens opérateurs dans une mesure avec blocage.
// ---------------------------------------------------------------------------------------------------------------------------
fn param_bloc(prec: Precision, mutant: u8) -> ParamGpu {
    let mut p = param(prec, mutant);
    p.bloc = ParamBloc { niveaux: vec![1, 2], ape_bloc: 1, lissage_bloc: 1, p_bloc: 0.7 };
    p
}

#[test]
#[ignore = "GPU"]
fn spectro_gpu_blocage_validation() {
    let Some(app) = appareil() else { return };
    for prec in [Precision::F64, Precision::F32] {
        let b = bilan_p(&app, &param_bloc(prec, 0));
        let tol = tolerance(prec);
        println!(
            "BLOCAGE {} : froid {:.2e} ; cpu rel max {:.3e} (norm {:.3e}) ; jauge {:.2e} ; translation {:.2e} ; cubique(48) {:.2e}",
            prec.nom(), b.froid, b.cpu_rel, b.cpu_norm, b.jauge, b.transl, b.cub
        );
        assert!(b.froid < tol * 10.0, "champ froid : {}", b.froid);
        assert!(b.cpu_norm < tol * 10.0, "contre le processeur : {}", b.cpu_norm);
        assert!(b.jauge < tol * 10.0, "jauge : {}", b.jauge);
        assert!(b.transl < tol * 10.0, "translation : {}", b.transl);
        assert!(b.cub < tol * 10.0, "cubique : {}", b.cub);
    }
}

/// Les opérateurs d'origine (niveaux APE) d'une mesure AVEC blocage sont identiques AU BIT à ceux d'une mesure sans blocage.
#[test]
#[ignore = "GPU"]
fn spectro_gpu_blocage_anciens_operateurs_au_bit() {
    let Some(app) = appareil() else { return };
    for prec in [Precision::F64, Precision::F32] {
        let p0 = param(prec, 0);
        let p1 = param_bloc(prec, 0);
        let mut s0 = SpectroGpu::nouveau(&app, &p0).unwrap();
        let mut s1 = SpectroGpu::nouveau(&app, &p1).unwrap();
        s0.init().unwrap();
        s0.avancer(30).unwrap();
        let cfgs: Vec<Vec<f64>> = (0..NCH).map(|c| s0.boite.telecharger(c).unwrap()).collect();
        s1.init().unwrap();
        for (c, u) in cfgs.iter().enumerate() {
            s1.boite.televerser(c, u).unwrap();
        }
        let r0 = s0.mesurer().unwrap();
        let r1 = s1.mesurer().unwrap();
        let (n0, n1) = (s0.nops(), s1.nops());
        let nt = p0.tau_max + 1;
        let paires = |n: usize| -> Vec<(usize, usize)> { (0..n).flat_map(|a| (a..n).map(move |b| (a, b))).collect() };
        let (pa0, pa1) = (paires(n0), paires(n1));
        let mut diff = 0usize;
        let mut total = 0usize;
        for c in 0..NCH {
            for (i0, pr) in pa0.iter().enumerate() {
                let i1 = pa1.iter().position(|x| x == pr).unwrap();
                for t in 0..nt {
                    total += 1;
                    if r0[c].corr[i0 * nt + t].to_bits() != r1[c].corr[i1 * nt + t].to_bits() {
                        diff += 1;
                    }
                    for k in 0..p0.ks.len() {
                        total += 1;
                        if r0[c].corr_ps[k][i0 * nt + t].to_bits() != r1[c].corr_ps[k][i1 * nt + t].to_bits() {
                            diff += 1;
                        }
                    }
                }
            }
            for o in 0..n0 {
                total += 1;
                if r0[c].moy[o].to_bits() != r1[c].moy[o].to_bits() {
                    diff += 1;
                }
            }
            if r0[c].plaq.to_bits() != r1[c].plaq.to_bits() {
                diff += 1;
            }
        }
        println!("BLOCAGE {} anciens opérateurs : {diff} valeurs différentes au bit sur {total}", prec.nom());
        assert_eq!(diff, 0);
    }
}

/// Mutants des noyaux de blocage (6 : agrafe basse sans dagger ; 7 : chemin direct décalé de 1 ; 8 : boucles à pas de 1).
#[test]
#[ignore = "GPU"]
fn spectro_gpu_blocage_mutants_detectes() {
    let Some(app) = appareil() else { return };
    for mutant in 6..=8u8 {
        let b = bilan_p(&app, &param_bloc(Precision::F64, mutant));
        let echec = b.froid > 1e-9 || b.cpu_norm > 1e-9 || b.jauge > 1e-9 || b.transl > 1e-9 || b.cub > 1e-9;
        println!(
            "MUTANT {mutant} : froid {:.1e} cpu {:.1e} jauge {:.1e} transl {:.1e} cubique {:.1e} -> {}",
            b.froid, b.cpu_norm, b.jauge, b.transl, b.cub, if echec { "DÉTECTÉ" } else { "NON DÉTECTÉ" }
        );
        assert!(echec, "mutant {mutant} non détecté");
    }
}

/// Torelons avec blocage (L_w = 8, superliens 2 et 4) contre `bloc::torelons_avec_blocs`, champ froid, jauge, translation.
#[test]
#[ignore = "GPU"]
#[allow(clippy::needless_range_loop)] // style seulement (indices explicites conservés : aucun changement numérique)
fn spectro_gpu_torelon_blocage() {
    let Some(app) = appareil() else { return };
    let g = GeoA::analyser("8,6,6,8").unwrap();
    for prec in [Precision::F64, Precision::F32] {
        let pb = ParamBloc { niveaux: vec![1, 2], ape_bloc: 2, lissage_bloc: 1, p_bloc: 0.6 };
        let p = ParamTorelon {
            cotes: [8, 6, 6, 8], w: 0, temps: 3, niveaux: vec![0, 2], alpha: 0.5, moments: vec![[0, 0], [1, 0]], tau_max: 4,
            precision: prec, nor: 4, betas: vec![2.3; NCH], graines: vec![31, 32, 33], bloc: pb.clone(), xi0: None, ops_bruts: false, profil: vec![], profil_niveau: None,
        };
        let mut tg = TorelonGpu::nouveau(&app, &p).unwrap();
        tg.init().unwrap();
        tg.avancer(40).unwrap();
        let res = tg.mesurer().unwrap();
        let ax = Axes::nouveaux(0, 3).unwrap();
        let (mut pire, mut pire_rel) = (0.0f64, 0.0f64);
        let mut base: Vec<Q> = Vec::new();
        for c in 0..NCH {
            let u = en_q(&tg.boite.telecharger(c).unwrap());
            if c == 0 {
                base = u.clone();
            }
            let (corr, pm) = bloc::torelons_avec_blocs(&g, &u, &ax, &p.niveaux, 0.5, &p.moments, 4, &pb).unwrap();
            for i in 0..p.moments.len() {
                let a: Vec<f64> = res[c].corr[i].iter().flatten().copied().collect();
                let b: Vec<f64> = corr[i].iter().flatten().copied().collect();
                let (rr, n) = ecart(&a, &b);
                pire = pire.max(n);
                pire_rel = pire_rel.max(rr);
                let a: Vec<f64> = res[c].phi_moyen[i].iter().flatten().copied().collect();
                let b: Vec<f64> = pm[i].iter().flatten().copied().collect();
                pire = pire.max(ecart(&a, &b).1);
            }
        }
        println!("TORELON-BLOC {} contre CPU : écart normalisé max {pire:.3e}, relatif max {pire_rel:.3e}", prec.nom());
        assert!(pire < tolerance(prec) * 10.0);
        let vec_corr = |m: &kvu::su2::spectro::gpu_torelon::MesTor| -> Vec<f64> { m.corr.iter().flatten().flatten().copied().collect() };
        let mut mesure = |u: &[Q]| -> Vec<f64> {
            tg.boite.televerser(0, &en_raw(u)).unwrap();
            vec_corr(&tg.mesurer().unwrap().remove(0))
        };
        let rv = mesure(&base);
        let mut uj = base.clone();
        transformer_jauge(&g, &mut uj, 99);
        let jg = ecart(&mesure(&uj), &rv).1;
        let tr = ecart(&mesure(&translation(&g, &base, [1, 2, 3, 5])), &rv).1;
        let froid = mesure(&vec![[1.0, 0.0, 0.0, 0.0]; g.v * 4]);
        // champ froid : C(p=0) = 1, C(p≠0) = 0 (paires × τ du moment 0 puis du moment 1)
        let nn = froid.len() / 2;
        let ef = froid[..nn].iter().fold(0.0f64, |m, x| m.max((x - 1.0).abs())).max(froid[nn..].iter().fold(0.0f64, |m, x| m.max(x.abs())));
        println!("TORELON-BLOC {} jauge {jg:.2e} translation(1,2,3,5) {tr:.2e} froid {ef:.2e}", prec.nom());
        assert!(jg < tolerance(prec) * 10.0 && tr < tolerance(prec) * 10.0 && ef < 1e-12);
    }
}

/// Impulsions hors axe : référence CPU, axe historique, invariance cubique et permutations de liste.
#[test]
#[ignore = "GPU"]
fn spectro_gpu_orbites_hors_axe() {
    let Some(app) = appareil() else { return };
    let orbites: Vec<_> = [[2, 2, 1], [3, 0, 0], [2, 2, 0], [2, 1, 1], [2, 1, 0]]
        .into_iter()
        .map(neutre::orbite)
        .collect();
    for (prec, avec_bloc) in [
        (Precision::F64, false), (Precision::F32, false),
        (Precision::F64, true), (Precision::F32, true),
    ] {
        let orbs = if avec_bloc { &orbites[..] } else { &orbites[..2] };
        let mut p = param(prec, 0);
        p.ks.clear();
        p.orbites = orbs.to_vec();
        if avec_bloc {
            p.bloc = ParamBloc { niveaux: vec![1, 2], ape_bloc: 1, lissage_bloc: 1, p_bloc: 1.0 };
            p.cotes = [8, 8, 8, 8];
        } else {
            // 343 sites par tranche, dont 87 dans le dernier bloc de 256.
            p.cotes = [7, 7, 7, 8];
        }
        let g = GeoA::nouvelle(p.cotes).unwrap();
        let tol = tolerance(prec);
        let mut sg = SpectroGpu::nouveau(&app, &p).unwrap();
        sg.init().unwrap();
        sg.avancer(40).unwrap();
        let cfgs: Vec<Vec<Q>> = (0..NCH).map(|c| en_q(&sg.boite.telecharger(c).unwrap())).collect();
        let res = sg.mesurer().unwrap();
        let formes: Vec<usize> = p.formes.iter().map(|n| FORMES.iter().position(|f| f.0 == n).unwrap()).collect();
        let pn = ParamNeutre { temps: p.temps, niveaux: p.niveaux.clone(), alpha: p.alpha, formes, tau_max: p.tau_max, mutant: MutantN::Aucun };
        let (mut rel, mut nor, mut pas_nul) = (0.0f64, 0.0f64, 0.0f64);
        for c in 0..NCH {
            let r = bloc::mesurer_avec_blocs_orb(&g, &cfgs[c], &pn, &[], orbs, &p.bloc).unwrap();
            for io in 0..orbs.len() {
                let rf: Vec<f64> = r.corr_orb[io].iter().flatten().copied().collect();
                let (a, b) = ecart(&res[c].corr_orb[io], &rf);
                rel = rel.max(a);
                nor = nor.max(b);
                pas_nul = pas_nul.max(rf.iter().fold(0.0f64, |m, x| m.max(x.abs())));
            }
            // l'impulsion nulle reste celle du processeur
            let rf: Vec<f64> = r.corr.iter().flatten().copied().collect();
            nor = nor.max(ecart(&res[c].corr, &rf).1);
        }
        assert!(pas_nul > 1e-6, "les corrélateurs d'orbite ne doivent pas être nuls");
        // champ froid
        let froid = mesure_champ(&mut sg, &vec![[1.0, 0.0, 0.0, 0.0]; g.v * 4]);
        let fr = froid.corr_orb.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
        // invariance cubique de la moyenne d'orbite (champ de la chaîne 0)
        let a = mesure_champ(&mut sg, &cfgs[0]);
        let mut cub = 0.0f64;
        for (perm, flip) in [([1, 2, 0], [false; 3]), ([0, 1, 2], [true, false, false]), ([2, 0, 1], [true, false, true])] {
            let b = mesure_champ(&mut sg, &cubique(&g, &cfgs[0], perm, flip));
            for io in 0..orbs.len() {
                cub = cub.max(ecart(&b.corr_orb[io], &a.corr_orb[io]).1);
            }
        }
        // Même champ, mêmes opérateurs et niveaux, avec l'ancien cran axial.
        let mut p_axe = p.clone();
        p_axe.orbites.clear();
        p_axe.ks = vec![3];
        let mut sg_axe = SpectroGpu::nouveau(&app, &p_axe).unwrap();
        sg_axe.init().unwrap();
        let axe = mesure_champ(&mut sg_axe, &cfgs[0]);
        let ea = ecart(&a.corr_orb[1], &axe.corr_ps[0]).1;
        assert!(a.corr.iter().zip(&axe.corr).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert!(a.moy.iter().zip(&axe.moy).all(|(x, y)| x.to_bits() == y.to_bits()));

        // L'ordre de sortie suit l'ordre de la liste. Changer la place d'une orbite
        // ou la quantité totale de vecteurs doit restituer les mêmes corrélateurs.
        let mut p_perm = p.clone();
        let permutation = if avec_bloc { vec![1, 0] } else { vec![4, 1, 3, 0, 2] };
        p_perm.orbites = permutation.iter().map(|&i| orbites[i].clone()).collect();
        let mut sg_perm = SpectroGpu::nouveau(&app, &p_perm).unwrap();
        sg_perm.init().unwrap();
        let mesures_perm = mesure_champ(&mut sg_perm, &cfgs[0]);
        assert_eq!(mesures_perm.corr_orb.len(), permutation.len());
        for (corr, i) in mesures_perm.corr_orb.iter().zip(permutation) {
            if i < orbs.len() {
                assert_eq!(corr.len(), a.corr_orb[i].len());
                assert!(corr.iter().zip(&a.corr_orb[i]).all(|(x, y)| x.to_bits() == y.to_bits()));
            }
        }

        println!("ORBITES {} bloc={avec_bloc} : cpu rel max {rel:.3e} (norm {nor:.3e}) ; froid {fr:.2e} ; cubique {cub:.2e} ; axe {ea:.2e}", prec.nom());
        assert!(nor < tol, "contre le processeur : {nor}");
        assert!(fr < tol, "champ froid : {fr}");
        assert!(cub < tol, "cubique : {cub}");
        assert!(ea < tol, "cran axial GPU : {ea}");
    }
}
