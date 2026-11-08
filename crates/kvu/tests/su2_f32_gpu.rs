//! Tests GPU du chemin FP32 (`#[ignore = "GPU"]`) et pilotes des gates de loi (LOI-0, LOI-1, chaînes pour
//! LOI-2/3/4/τ, LOT-1-F32, REPRISE-1-F32, banc). Lancement : `validation/f32/gates_f32.sh`, qui prend le
//! verrou GPU de la machine PAR FENÊTRE de 10 minutes puis positionne `KVU_F32_SOUS_VERROU=1`.
//! Sans cette variable, ou sans GPU, chaque test imprime `SKIP <nom>` (qui ne vaut PAS vert) et rend la main.
//!
//! Sorties des pilotes : dans `$KVU_LOI_OUT` (répertoire de travail sous
//! /chemin/vers/projet/kevinotron_dev/scratch/f32, nettoyé à la fin), jugées par `validation/f32/juge_f32.py`.
//! Chaque gate calculée ici imprime `PASS <nom>` ou `FAIL <nom>`.

use kvu::commun::{BackendKind, ChampHote, Depart, Graine, Resultat};
use kvu::gpu::Appareil;
use kvu::heritage::su2_hb::{g17, measure_quat, Geo, Q};
use kvu::su2::rapide32::{tirage_x0_hote, Rapide32Gpu};
use kvu::su2::{Gel, NoyauxSu2, ParamChaine};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

// ------------------------------------------------------------------------------------------
// Aides
// ------------------------------------------------------------------------------------------

/// GPU de la machine, ou None (avec `SKIP`) : pas sous verrou, pas de GPU, ou erreur d'ouverture.
fn appareil(gate: &str) -> Option<Appareil> {
    if std::env::var("KVU_F32_SOUS_VERROU").as_deref() != Ok("1") {
        println!("SKIP {gate} (pas sous verrou GPU : lancer par validation/f32/gates_f32.sh)");
        return None;
    }
    // même choix de backend que les gates de T2 (gpu_runtime.rs) : KEVINOTRON_BACKEND, sinon CUDA si la feature
    // `cuda` est compilée (gamer), sinon HIP (acemagic). `ouvrir_auto` (détection de machine, T1) en est la version finale.
    let backend = match std::env::var("KEVINOTRON_BACKEND").ok().and_then(|v| v.parse().ok()) {
        Some(b) if b != BackendKind::Cpu => b,
        _ if cfg!(feature = "cuda") => BackendKind::Cuda,
        _ => BackendKind::Hip,
    };
    match kvu::gpu::ouvrir(backend, 0) {
        Ok(app) => {
            println!("# {gate} : {} / {} / {}", app.backend().nom(), app.nom(), app.arch());
            Some(app)
        }
        Err(e) => {
            println!("SKIP {gate} (backend {} indisponible : {e})", backend.nom());
            None
        }
    }
}

fn env_ou(nom: &str, defaut: &str) -> String {
    std::env::var(nom).unwrap_or_else(|_| defaut.to_string())
}

fn env_u64(nom: &str, defaut: u64) -> u64 {
    env_ou(nom, &defaut.to_string()).parse().unwrap_or_else(|_| panic!("{nom} : entier attendu"))
}

/// `$KVU_LOI_OUT/<sous>` (créé), ou None avec un message si KVU_LOI_OUT est absent.
fn sortie(gate: &str, sous: &str) -> Option<PathBuf> {
    let Ok(base) = std::env::var("KVU_LOI_OUT") else {
        println!("SKIP {gate} (KVU_LOI_OUT absent : aucune sortie à écrire)");
        return None;
    };
    let d = Path::new(&base).join(sous);
    std::fs::create_dir_all(&d).unwrap_or_else(|e| panic!("création de {} : {e}", d.display()));
    Some(d)
}

/// Garde SSD : refuse d'écrire si le libre du volume de `chemin` passerait sous le seuil
/// (`KVU_F32_SSD_MIN_GO`, défaut 110 Go ; gamer : 50).
fn garde_ssd(chemin: &Path, octets: u64) {
    let seuil: f64 = env_ou("KVU_F32_SSD_MIN_GO", "110").parse().expect("KVU_F32_SSD_MIN_GO");
    let sortie = std::process::Command::new("df")
        .args(["--output=avail", "-B1"])
        .arg(chemin)
        .output()
        .expect("df");
    let libre: f64 = String::from_utf8_lossy(&sortie.stdout).lines().nth(1).unwrap_or("0").trim().parse().unwrap_or(0.0);
    let libre_apres_go = (libre - octets as f64) / 1e9;
    assert!(libre_apres_go >= seuil, "garde SSD : {libre_apres_go:.1} Go libres après écriture < seuil {seuil} Go : REFUS");
}

fn chaine(beta: f64, graine: u64, depart: Depart) -> ParamChaine {
    ParamChaine { beta, graine: Graine(graine), depart }
}

fn nouveau(app: &Appareil, l: usize, chaines: Vec<ParamChaine>) -> Rapide32Gpu {
    Rapide32Gpu::nouveau(app, Rapide32Gpu::config(l, chaines, 4)).expect("Rapide32Gpu::nouveau")
}

/// Un pas = 1 sweep de bain thermique + `nor` sweeps d'overrelaxation.
fn pas(n: &mut Rapide32Gpu) {
    n.sweep_hb().expect("hb");
    for _ in 0..n.cfg.nor {
        n.sweep_or().expect("or");
    }
}

fn champ_en_q(h: &ChampHote) -> Vec<Q> {
    h.q.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn ecrire_json(chemin: &Path, v: &serde_json::Value) {
    std::fs::write(chemin, serde_json::to_string_pretty(v).expect("json")).unwrap_or_else(|e| panic!("{}: {e}", chemin.display()));
}

fn verdict(nom: &str, ok: bool) {
    println!("{} {nom}", if ok { "PASS" } else { "FAIL" });
}

// ------------------------------------------------------------------------------------------
// Tests unitaires GPU
// ------------------------------------------------------------------------------------------

#[test]
#[ignore = "GPU"]
fn f32_demarrage_froid_et_chaud() {
    let Some(app) = appareil("F32-DEMARRAGE") else { return };
    let mut n = nouveau(&app, 6, vec![chaine(2.3, 11, Depart::Chaud), chaine(2.3, 12, Depart::Froid), chaine(2.3, 13, Depart::Chaud)]);
    n.init().unwrap();
    n.allouer_historique(2).unwrap();
    n.mesurer(0).unwrap();
    let h = n.historique().unwrap();
    // chaîne froide : toutes les plaquettes et tous les Polyakov valent exactement 1
    assert_eq!(&h[10..15], &[1.0; 5], "départ froid : P = P0..P3 = 1 exactement");
    // chaînes chaudes : Haar, moyenne nulle (écart-type de la plaquette moyenne ≈ 0,5/√(6V) = 0,006)
    for c in [0usize, 2] {
        let r = &h[10 * c..10 * c + 5];
        assert!(r[0].abs() < 0.05, "chaîne {c} : plaquette de Haar = {}", r[0]);
        assert!(r[1..].iter().all(|p| p.abs() < 0.2), "chaîne {c} : Polyakov de Haar = {:?}", &r[1..]);
        assert!(n.telecharger(c).unwrap().defaut_unitarite() < 2e-7, "chaîne {c} : champ de Haar non unitaire");
    }
    assert!(h[0] != h[10 * 2], "deux graines : deux champs");
    assert_eq!(n.telecharger(1).unwrap().defaut_unitarite(), 0.0);
    verdict("F32-DEMARRAGE", true);
}

#[test]
#[ignore = "GPU"]
fn f32_mesures_egales_au_calcul_hote() {
    let Some(app) = appareil("F32-MESURES") else { return };
    for l in [4usize, 6, 8] {
        let mut n = nouveau(&app, l, vec![chaine(2.3, 21, Depart::Chaud), chaine(8.0, 22, Depart::Froid)]);
        n.init().unwrap();
        for _ in 0..5 {
            pas(&mut n);
        }
        n.allouer_historique(1).unwrap();
        n.mesurer(0).unwrap();
        n.mesurer_tranches(0, 1).unwrap();
        let (h, tr) = (n.historique().unwrap(), n.historique_tranches().unwrap());
        let g = Geo::new(l);
        let v = l.pow(4);
        for c in 0..2 {
            let q = champ_en_q(&n.telecharger(c).unwrap());
            let attendu = measure_quat(&g, &q);
            let ecart = (0..5).map(|k| (h[5 * c + k] - attendu[k]).abs()).fold(0.0, f64::max);
            assert!(ecart <= 1e-13, "L = {l}, chaîne {c} : mesures device/hôte à {ecart:e}");
            // tranches : S_t (plans 12, 13, 23), T_t (plans 01, 02, 03), 1/2 ReTr, moyenne sur le site (3 plaquettes)
            let l3 = v / l;
            let mut ec_tr = 0.0f64;
            for t in 0..l {
                let (mut s_t, mut t_t) = (0.0, 0.0);
                for s in t * l3..(t + 1) * l3 {
                    let x = g.coords(s);
                    for mu in 0..4 {
                        let smu = g.nb_up(s, mu, x[mu]);
                        for (nu, &xnu) in x.iter().enumerate().skip(mu + 1) {
                            let snu = g.nb_up(s, nu, xnu);
                            let a = |m: usize, si: usize| q[m * v + si];
                            let p = kvu::heritage::su2_hb::qmul(
                                kvu::heritage::su2_hb::qmul(a(mu, s), a(nu, smu)),
                                kvu::heritage::su2_hb::qmul(kvu::heritage::su2_hb::qdag(a(mu, snu)), kvu::heritage::su2_hb::qdag(a(nu, s))),
                            );
                            if mu == 0 {
                                t_t += p[0];
                            } else {
                                s_t += p[0];
                            }
                        }
                    }
                }
                let sc = 1.0 / (3.0 * l3 as f64);
                ec_tr = ec_tr.max((tr[2 * l * c + t] - s_t * sc).abs()).max((tr[2 * l * c + l + t] - t_t * sc).abs());
            }
            assert!(ec_tr <= 1e-13, "L = {l}, chaîne {c} : tranches device/hôte à {ec_tr:e}");
        }
    }
    verdict("F32-MESURES", true);
}

#[test]
#[ignore = "GPU"]
fn f32_gel_du_bord_respecte() {
    let Some(app) = appareil("F32-GEL") else { return };
    let l = 4usize;
    let mut n = nouveau(&app, l, vec![chaine(2.8, 31, Depart::Chaud), chaine(2.8, 32, Depart::Chaud)]);
    n.init().unwrap();
    n.fixer_gel(Gel::BordX0Spatial).unwrap();
    let avant = n.etat_brut().unwrap();
    for _ in 0..4 {
        pas(&mut n);
    }
    let apres = n.etat_brut().unwrap();
    let v = l.pow(4);
    let (mut gelés, mut changes) = (0usize, 0usize);
    for c in 0..2 {
        for mu in 0..4 {
            for s in 0..v {
                let (x0, i) = (s / (l * l * l), ((c * 4 + mu) * v + s) * 4);
                let egal = bits(&avant[i..i + 4]) == bits(&apres[i..i + 4]);
                if mu != 0 && x0 == 0 {
                    assert!(egal, "lien gelé modifié : chaîne {c}, mu {mu}, site {s}");
                    gelés += 1;
                } else if !egal {
                    changes += 1;
                }
            }
        }
    }
    assert_eq!(gelés, 2 * 3 * l * l * l, "nombre de liens gelés");
    assert!(changes > 2 * (4 * v - 3 * l * l * l) * 9 / 10, "les liens non gelés doivent évoluer ({changes} changés)");
    // sans gel, les liens du bord évoluent aussi
    n.fixer_gel(Gel::Aucun).unwrap();
    let avant = apres;
    pas(&mut n);
    let apres = n.etat_brut().unwrap();
    let bord_change = (0..2 * 4 * v).filter(|&k| !(k / v).is_multiple_of(4) && (k % v) / (l * l * l) == 0).any(|k| bits(&avant[4 * k..4 * k + 4]) != bits(&apres[4 * k..4 * k + 4]));
    assert!(bord_change, "sans gel, le bord doit évoluer");
    verdict("F32-GEL", true);
}

#[test]
#[ignore = "GPU"]
fn f32_tirage_noyau_accord_avec_le_miroir_hote() {
    let Some(app) = appareil("F32-MIROIR-X0") else { return };
    let n = nouveau(&app, 4, vec![chaine(2.3, 1, Depart::Chaud)]);
    for (alpha, graine) in [(0.3f64, 5u64), (1.999, 6), (2.0, 7), (5.0, 8), (100.0, 9)] {
        const N: usize = 50_000;
        let (g, plafond) = n.test_x0(alpha, N, graine).unwrap();
        assert_eq!(plafond, 0);
        let (k0, k1) = Graine(graine).cle();
        let mut divergents = 0usize;
        for (i, &x) in g.iter().enumerate() {
            let mut draw = 0;
            let (h, _, _) = tirage_x0_hote(alpha as f32, i as u32, 0x7E57, &mut draw, k0, k1);
            // mêmes mots Philox : mêmes décisions d'acceptation, valeurs égales à quelques ULP de la libm près
            if (x - f64::from(h)).abs() > 2e-5 {
                divergents += 1;
            }
        }
        println!("accord x0 α = {alpha} : {divergents} tirages divergents sur {N}");
        assert!(divergents * 5000 <= N, "α = {alpha} : {divergents}/{N} tirages divergents (> 0,02 %)");
    }
    verdict("F32-MIROIR-X0", true);
}

#[test]
#[ignore = "GPU"]
fn f32_compteurs_et_essais() {
    let Some(app) = appareil("F32-COMPTEURS") else { return };
    let mut n = nouveau(&app, 6, vec![chaine(2.3, 41, Depart::Chaud), chaine(16.0, 42, Depart::Chaud)]);
    n.init().unwrap();
    for _ in 0..10 {
        n.sweep_hb().unwrap();
    }
    let c = n.compteurs().unwrap();
    let liens = 10.0 * 4.0 * 6f64.powi(4);
    for (k, ck) in c.iter().enumerate() {
        let r = ck.essais as f64 / liens;
        println!("chaîne {k} : essais par lien = {r:.4}, plafond = {}", ck.plafond);
        assert_eq!(ck.plafond, 0);
        assert!((1.0..3.0).contains(&r), "essais par lien {r}");
    }
    assert_eq!(n.sweep_id(), 10);
    verdict("F32-COMPTEURS", true);
}

#[test]
#[ignore = "GPU"]
fn f32_telecharger_televerser_aller_retour_exact() {
    let Some(app) = appareil("F32-ALLER-RETOUR") else { return };
    let mut n = nouveau(&app, 4, vec![chaine(2.3, 51, Depart::Chaud), chaine(2.3, 52, Depart::Chaud)]);
    n.init().unwrap();
    pas(&mut n);
    let brut = n.etat_brut().unwrap();
    let h = n.telecharger(1).unwrap();
    assert_eq!(h.q.len(), 16 * 256);
    assert!(h.defaut_unitarite() < 2e-7);
    // téléverser dans l'autre chaîne puis relire : mêmes bits
    n.televerser(0, &h).unwrap();
    let apres = n.etat_brut().unwrap();
    assert_eq!(bits(&apres[..16 * 256]), bits(&brut[16 * 256..]), "téléversement d'un champ issu du device : bits identiques");
    assert_eq!(bits(&apres[16 * 256..]), bits(&brut[16 * 256..]), "l'autre chaîne n'est pas touchée");
    // un champ hôte hors sphère est normalisé
    let mut h2 = h.clone();
    for c in h2.q.chunks_exact_mut(4) {
        for v in c {
            *v *= 3.0;
        }
    }
    n.televerser(0, &h2).unwrap();
    assert!(n.telecharger(0).unwrap().defaut_unitarite() < 2e-7, "champ de norme 3 normalisé au téléversement");
    // erreurs
    assert!(n.televerser(2, &h).is_err() && n.telecharger(2).is_err(), "chaîne hors lot");
    let faux = ChampHote::froid(6).unwrap();
    assert!(n.televerser(0, &faux).is_err(), "L différent refusé");
    verdict("F32-ALLER-RETOUR", true);
}

#[test]
#[ignore = "GPU"]
fn f32_determinisme_et_graine() {
    let Some(app) = appareil("F32-DETERMINISME") else { return };
    let course = |graine: u64| -> Vec<u32> {
        let mut n = nouveau(&app, 6, vec![chaine(2.8, graine, Depart::Chaud)]);
        n.init().unwrap();
        for _ in 0..20 {
            pas(&mut n);
        }
        bits(&n.etat_brut().unwrap())
    };
    let (a, b, c) = (course(61), course(61), course(62));
    assert_eq!(a, b, "double run identique à l'octet");
    assert_ne!(a, c, "graine + 1 ⇒ champ différent");
    verdict("F32-DETERMINISME", true);
}

#[test]
#[ignore = "GPU"]
fn f32_provenance_et_mutant() {
    let Some(app) = appareil("F32-PROVENANCE") else { return };
    let normal = nouveau(&app, 4, vec![chaine(2.3, 1, Depart::Chaud)]);
    let mut cfg = Rapide32Gpu::config(4, vec![chaine(2.3, 1, Depart::Chaud)], 4);
    cfg.mutant_metro = true;
    let mutant = Rapide32Gpu::nouveau(&app, cfg).unwrap();
    let (p, m) = (&normal.provenance()[0], &mutant.provenance()[0]);
    assert_eq!(p.nom, "su2_f32");
    assert_eq!(p.sha256_source.len(), 64);
    assert_eq!(p.sha256_code.len(), 64);
    assert_eq!(p.parties.len(), 3, "prélude + su2_commun.h + su2_f32.cu : {:?}", p.parties);
    assert_ne!(p.sha256_source, m.sha256_source, "le mutant change la source assemblée");
    assert!(!p.arguments.is_empty());
    verdict("F32-PROVENANCE", true);
}

#[test]
#[ignore = "GPU"]
fn f32_mutant_metro_rouge_au_test_x0() {
    let Some(app) = appareil("F32-MUTANT-X0") else { return };
    let mut cfg = Rapide32Gpu::config(4, vec![chaine(2.3, 1, Depart::Chaud)], 4);
    cfg.mutant_metro = true;
    let n = Rapide32Gpu::nouveau(&app, cfg).unwrap();
    // depuis x = 1 : le Metropolis naïf accepte toujours (pc = 0) ⇒ tirage UNIFORME sur [-1,1] : moyenne ≈ 0
    let (x, plafond) = n.test_x0(100.0, 100_000, 3).unwrap();
    assert_eq!(plafond, 0);
    let m1 = x.iter().sum::<f64>() / x.len() as f64;
    assert!(m1.abs() < 0.02, "mutant : moyenne {m1} (uniforme attendu) au lieu de ≈ 0,985 pour α = 100");
    verdict("F32-MUTANT-X0", true);
}

// ------------------------------------------------------------------------------------------
// Gates : LOT-1-F32, REPRISE-1-F32
// ------------------------------------------------------------------------------------------

/// Joue `nsteps` pas avec mesure à chaque pas ; renvoie (champ brut, historique, plafond total).
fn jouer(n: &mut Rapide32Gpu, nsteps: usize) -> Resultat<(Vec<f32>, Vec<f64>, u64)> {
    n.allouer_historique(nsteps + 1)?;
    n.mesurer(0)?;
    for t in 1..=nsteps {
        pas(n);
        n.mesurer(t)?;
    }
    let plafond = n.compteurs()?.iter().map(|c| c.plafond).sum();
    Ok((n.etat_brut()?, n.historique()?, plafond))
}

/// LOT-1-F32 : la chaîne c d'un lot de 4 donne les mêmes octets que la même chaîne lancée seule.
#[test]
#[ignore = "GPU"]
fn lot_1_f32() {
    let Some(app) = appareil("LOT-1-F32") else { return };
    const L: usize = 6;
    const N: usize = 120;
    let specs = [
        chaine(2.3, 71, Depart::Chaud),
        chaine(2.8, 72, Depart::Froid),
        chaine(3.5, 73, Depart::Chaud),
        chaine(8.0, 0x1_0000_0000 + 74, Depart::Chaud), // graine > 32 bits : clé haute non nulle
    ];
    let mut lot = nouveau(&app, L, specs.to_vec());
    lot.init().unwrap();
    let (champ, hist, plafond_lot) = jouer(&mut lot, N).unwrap();
    let (v16, cap) = (16 * L.pow(4), N + 1);
    let mut details = Vec::new();
    let mut ok = plafond_lot == 0;
    for (c, sp) in specs.iter().enumerate() {
        let mut seul = nouveau(&app, L, vec![sp.clone()]);
        seul.init().unwrap();
        let (champ_s, hist_s, plafond_s) = jouer(&mut seul, N).unwrap();
        let champ_ok = bits(&champ[c * v16..(c + 1) * v16]) == bits(&champ_s);
        let hist_ok = hist[c * cap * 5..(c + 1) * cap * 5].iter().map(|x| x.to_bits()).eq(hist_s.iter().map(|x| x.to_bits()));
        println!("LOT-1-F32 chaîne {c} β = {} : champ {champ_ok}, historique {hist_ok}, plafond {plafond_s}", sp.beta);
        details.push(format!("c{c}: champ={champ_ok} hist={hist_ok}"));
        ok &= champ_ok && hist_ok && plafond_s == 0;
    }
    if let Some(d) = sortie("LOT-1-F32", ".") {
        ecrire_json(&d.join("lot1.json"), &serde_json::json!({"ok": ok, "detail": details.join("; "), "pas": N, "l": L}));
    }
    verdict("LOT-1-F32", ok);
    assert!(ok, "LOT-1-F32 ROUGE : {details:?}");
}

fn config_reprise() -> (usize, Vec<ParamChaine>, usize) {
    (6, vec![chaine(2.3, 81, Depart::Chaud), chaine(2.8, 82, Depart::Chaud), chaine(3.5, 83, Depart::Froid)], 150)
}

fn lire_f32(chemin: &Path) -> Vec<f32> {
    std::fs::read(chemin).expect("lecture").chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn lire_f64(chemin: &Path) -> Vec<f64> {
    std::fs::read(chemin).expect("lecture").chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect()
}

fn ecrire_octets<T: Copy>(chemin: &Path, v: &[T], conv: impl Fn(T) -> Vec<u8>) {
    let mut o = Vec::new();
    for &x in v {
        o.extend(conv(x));
    }
    std::fs::write(chemin, o).expect("écriture");
}

/// Phase 1 de la reprise (nouveau processus) : N pas depuis le départ, état sauvé dans $KVU_F32_PHASE_DIR.
#[test]
#[ignore = "GPU"]
fn reprise_phase_a() {
    let Ok(dir) = std::env::var("KVU_F32_PHASE_DIR") else { return };
    let Some(app) = appareil("REPRISE-PHASE-A") else { std::process::exit(3) };
    let dir = PathBuf::from(dir);
    let (l, chaines, n) = config_reprise();
    let mut g = nouveau(&app, l, chaines);
    g.init().unwrap();
    let (champ, hist, _) = jouer(&mut g, n).unwrap();
    ecrire_octets(&dir.join("etat.f32"), &champ, |x| x.to_le_bytes().to_vec());
    ecrire_octets(&dir.join("hist_a.f64"), &hist, |x| x.to_le_bytes().to_vec());
    std::fs::write(dir.join("sweep.txt"), g.sweep_id().to_string()).expect("sweep");
}

/// Phase 2 (autre nouveau processus) : relit l'état, joue N pas de plus.
#[test]
#[ignore = "GPU"]
fn reprise_phase_b() {
    let Ok(dir) = std::env::var("KVU_F32_PHASE_DIR") else { return };
    let Some(app) = appareil("REPRISE-PHASE-B") else { std::process::exit(3) };
    let dir = PathBuf::from(dir);
    let (l, chaines, n) = config_reprise();
    let mut g = nouveau(&app, l, chaines);
    let sweep: u32 = std::fs::read_to_string(dir.join("sweep.txt")).expect("sweep").trim().parse().expect("sweep");
    g.restaurer_brut(&lire_f32(&dir.join("etat.f32")), sweep).unwrap();
    let (champ, hist, _) = jouer(&mut g, n).unwrap();
    ecrire_octets(&dir.join("etat_b.f32"), &champ, |x| x.to_le_bytes().to_vec());
    ecrire_octets(&dir.join("hist_b.f64"), &hist, |x| x.to_le_bytes().to_vec());
    std::fs::write(dir.join("sweep_b.txt"), g.sweep_id().to_string()).expect("sweep");
}

fn lancer_phase(nom: &str, dir: &Path) -> bool {
    let st = std::process::Command::new(std::env::current_exe().expect("exe"))
        .args(["--ignored", "--exact", nom, "--nocapture", "--test-threads=1"])
        .env("KVU_F32_PHASE_DIR", dir)
        .status()
        .expect("lancement du processus fils");
    st.success()
}

/// REPRISE-1-F32 : 2N pas d'un coup = N pas, puis reprise DANS UN NOUVEAU PROCESSUS, puis N pas, à l'octet.
#[test]
#[ignore = "GPU"]
fn reprise_1_f32() {
    if std::env::var("KVU_F32_PHASE_DIR").is_ok() {
        return; // appelé comme fils : sans objet
    }
    let (l, chaines, n) = config_reprise();
    let (champ_a, hist_a, plafond_a) = {
        let Some(app) = appareil("REPRISE-1-F32") else { return };
        let mut g = nouveau(&app, l, chaines.clone());
        g.init().unwrap();
        jouer(&mut g, 2 * n).unwrap()
    }; // l'appareil du parent est rendu avant de lancer les fils
    let dir = std::env::temp_dir().join(format!("kvu_f32_reprise_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("répertoire de reprise");
    let ok_fils = lancer_phase("reprise_phase_a", &dir) && lancer_phase("reprise_phase_b", &dir);
    let mut ok = ok_fils && plafond_a == 0;
    let mut detail = format!("fils ok = {ok_fils}");
    if ok_fils {
        let (champ_b, hist_1, hist_2) = (lire_f32(&dir.join("etat_b.f32")), lire_f64(&dir.join("hist_a.f64")), lire_f64(&dir.join("hist_b.f64")));
        let sweep_b: u32 = std::fs::read_to_string(dir.join("sweep_b.txt")).expect("sweep_b").trim().parse().expect("u32");
        let champ_ok = bits(&champ_a) == bits(&champ_b);
        // historique : phase 1 = cases 0..=N de A ; phase 2 = cases N..=2N de A (case 0 de la phase 2 = état repris)
        let (cap_a, cap_1, nch) = (2 * n + 1, n + 1, 3);
        let mut hist_ok = true;
        for c in 0..nch {
            for j in 0..=n {
                for k in 0..5 {
                    let a1 = hist_a[(c * cap_a + j) * 5 + k].to_bits();
                    let a2 = hist_a[(c * cap_a + n + j) * 5 + k].to_bits();
                    hist_ok &= a1 == hist_1[(c * cap_1 + j) * 5 + k].to_bits() && a2 == hist_2[(c * cap_1 + j) * 5 + k].to_bits();
                }
            }
        }
        let sweep_ok = sweep_b == 2 * n as u32;
        detail = format!("champ {champ_ok}, historique {hist_ok}, sweep_id {sweep_ok}, fils ok");
        ok &= champ_ok && hist_ok && sweep_ok;
    }
    let _ = std::fs::remove_dir_all(&dir);
    if let Some(d) = sortie("REPRISE-1-F32", ".") {
        ecrire_json(&d.join("reprise.json"), &serde_json::json!({"ok": ok, "detail": detail, "n": n, "l": l}));
    }
    println!("REPRISE-1-F32 : {detail}");
    verdict("REPRISE-1-F32", ok);
    assert!(ok, "REPRISE-1-F32 ROUGE : {detail}");
}

// ------------------------------------------------------------------------------------------
// Pilotes de gates de loi : LOI-0, LOI-1 (+ mutant), chaînes (LOI-2, LOI-3, LOI-4a, LOI-τ), banc
// ------------------------------------------------------------------------------------------

/// LOI-0 : après 10⁴ pas à L = 8, β = 2,3 : max | |q| − 1 | ≤ 2e-6 (pas de dérive), plafond = 0.
#[test]
#[ignore = "GPU"]
fn loi_0_pilote() {
    let Some(app) = appareil("LOI-0") else { return };
    let (nsteps, controle) = (env_u64("KVU_F32_LOI0_PAS", 10_000) as usize, 1000usize);
    let mut n = nouveau(&app, 8, vec![chaine(2.3, 20_261_004_001, Depart::Chaud)]);
    n.init().unwrap();
    let mut defauts = Vec::new();
    for t in 1..=nsteps {
        pas(&mut n);
        if t % controle == 0 || t == nsteps {
            defauts.push(n.telecharger(0).unwrap().defaut_unitarite());
        }
    }
    let plafond: u64 = n.compteurs().unwrap().iter().map(|c| c.plafond).sum();
    let (final_, max) = (*defauts.last().unwrap(), defauts.iter().copied().fold(0.0, f64::max));
    println!("LOI-0 : L = 8, β = 2,3, {nsteps} pas : défaut de norme final {final_:.3e}, max {max:.3e}, plafond {plafond}");
    if let Some(d) = sortie("LOI-0", "loi0") {
        ecrire_json(
            &d.join("loi0.json"),
            &serde_json::json!({"l": 8, "beta": 2.3, "pas": nsteps, "defauts": defauts, "defaut_final": final_, "defaut_max": max, "plafond": plafond}),
        );
    }
    verdict("LOI-0 (pilote)", final_ <= 2e-6 && max <= 2e-6 && plafond == 0);
    assert!(final_ <= 2e-6 && max <= 2e-6 && plafond == 0);
}

/// Écrit les tirages de x0 des 7 α dans `$KVU_LOI_OUT/loi1/<sous>` : moments (sommes en double) et 4·10⁵ tirages pour le KS.
fn pilote_loi1(gate: &str, sous: &str, mutant: bool, graine0: u64) {
    let Some(app) = appareil(gate) else { return };
    let Some(d) = sortie(gate, &format!("loi1/{sous}")) else { return };
    let n_tirages = env_u64("KVU_F32_LOI1_N", 20_000_000) as usize;
    let mut cfg = Rapide32Gpu::config(4, vec![chaine(2.3, 1, Depart::Chaud)], 4);
    cfg.mutant_metro = mutant;
    let g = Rapide32Gpu::nouveau(&app, cfg).expect("nouveau");
    let mut lignes = String::new();
    for (i, &alpha) in [0.3f64, 1.0, 1.999, 2.0, 5.0, 20.0, 100.0].iter().enumerate() {
        let graine = graine0 + i as u64;
        let (x, plafond) = g.test_x0(alpha, n_tirages, graine).expect("test_x0");
        let nf = x.len() as f64;
        let (mut m1, mut m2, mut mn, mut mx) = (0.0f64, 0.0f64, f64::MAX, f64::MIN);
        for &v in &x {
            m1 += v;
            m2 += v * v;
            mn = mn.min(v);
            mx = mx.max(v);
        }
        let fichier = format!("ks_{i}.f32");
        ecrire_octets(&d.join(&fichier), &x[..400_000.min(x.len())], |v| (v as f32).to_le_bytes().to_vec());
        let alpha_f32 = f64::from(alpha as f32);
        let ligne = serde_json::json!({
            "alpha_demande": alpha, "alpha": alpha_f32, "n": x.len(), "m1": m1 / nf, "m2": m2 / nf,
            "plafond": plafond, "min": mn, "max": mx, "graine": graine, "ks_fichier": fichier, "mutant": mutant
        });
        println!("{gate} α = {alpha_f32:<10} m1 = {} m2 = {} plafond = {plafond}", g17(m1 / nf), g17(m2 / nf));
        writeln!(lignes, "{ligne}").expect("écriture en mémoire");
    }
    std::fs::write(d.join("res.jsonl"), lignes).expect("res.jsonl");
}

/// LOI-1 : tirages exacts de x0 (nominal) pour le juge Python (moments |z| < 4, KS p > 1e-4).
#[test]
#[ignore = "GPU"]
fn loi_1_pilote() {
    pilote_loi1("LOI-1", "nominal", false, 9000);
}

/// LOI-4 (b) : le mutant Metropolis doit être ROUGE à LOI-1 pour α ≥ 5 (jugé par Python).
#[test]
#[ignore = "GPU"]
fn loi_4b_pilote_mutant_metro() {
    pilote_loi1("LOI-4b", "mutant", true, 9100);
}

/// Ensemble de chaînes d'un lot au format de su2_hip (therm.csv, measurements.csv, manifest.json, COMPLETE,
/// cfg_NNNNNN.q64), pour l'analyse de analyse_plaquette.py.
#[allow(clippy::too_many_arguments)]
fn ecrire_lot(
    app: &Appareil,
    l: usize,
    nor: usize,
    therm: usize,
    nmeas: usize,
    dump_every: usize,
    racine: &Path,
    chaines: &[(String, String, ParamChaine)], // (étiquette de cellule « L6_b2.3_c1 », β nominal, paramètres)
) {
    let params: Vec<ParamChaine> = chaines.iter().map(|c| c.2.clone()).collect();
    let mut cfg = Rapide32Gpu::config(l, params, nor);
    cfg.mutant_metro = false;
    let mut n = Rapide32Gpu::nouveau(app, cfg).expect("nouveau");
    let nch = chaines.len();
    let cap = therm.max(nmeas) + 1;
    n.init().unwrap();
    n.allouer_historique(cap).unwrap();
    n.mesurer(0).unwrap();
    let t0 = std::time::Instant::now();
    for t in 1..=therm {
        pas(&mut n);
        n.mesurer(t).unwrap();
    }
    let h = n.historique().unwrap();
    let mut therm_csv: Vec<String> = vec!["step,plaquette,P0,P1,P2,P3\n".to_string(); nch];
    for (c, s) in therm_csv.iter_mut().enumerate() {
        for i in 0..=therm {
            let r = &h[(c * cap + i) * 5..(c * cap + i) * 5 + 5];
            writeln!(s, "{i},{},{},{},{},{}", g17(r[0]), g17(r[1]), g17(r[2]), g17(r[3]), g17(r[4])).expect("mem");
        }
    }
    let t_therm = t0.elapsed().as_secs_f64();
    // production : la case 0 est l'état de fin de thermalisation
    n.mesurer(0).unwrap();
    let dossiers: Vec<PathBuf> = chaines.iter().map(|c| racine.join(&c.0)).collect();
    for d in &dossiers {
        std::fs::create_dir_all(d).unwrap_or_else(|e| panic!("{}: {e}", d.display()));
    }
    let octets_dump = nmeas.checked_div(dump_every).map_or(0, |k| (nch * 8 * 16 * l.pow(4) * (k + 1)) as u64);
    garde_ssd(racine, octets_dump + 100_000_000);
    for m in 1..=nmeas {
        pas(&mut n);
        n.mesurer(m).unwrap();
        if dump_every > 0 && m % dump_every == 0 {
            for (c, d) in dossiers.iter().enumerate() {
                let champ = n.telecharger(c).unwrap();
                ecrire_octets(&d.join(format!("cfg_{m:06}.q64")), &champ.q, |x| x.to_le_bytes().to_vec());
            }
        }
    }
    let h = n.historique().unwrap();
    let compteurs = n.compteurs().unwrap();
    let prov = n.provenance();
    let wall = t0.elapsed().as_secs_f64();
    let plafond_total: u64 = compteurs.iter().map(|c| c.plafond).sum();
    for (c, d) in dossiers.iter().enumerate() {
        let mut s = String::from("measurement_index,total_steps,phase,plaquette,P0,P1,P2,P3\n");
        for i in 0..=nmeas {
            let r = &h[(c * cap + i) * 5..(c * cap + i) * 5 + 5];
            writeln!(s, "{i},{},{},{},{},{},{},{}", therm + i, if i == 0 { "initial" } else { "sample" }, g17(r[0]), g17(r[1]), g17(r[2]), g17(r[3]), g17(r[4])).expect("mem");
        }
        std::fs::write(d.join("measurements.csv"), s).expect("measurements.csv");
        std::fs::write(d.join("therm.csv"), &therm_csv[c]).expect("therm.csv");
        let p = &chaines[c].2;
        let essais_par_lien = compteurs[c].essais as f64 / (4.0 * l.pow(4) as f64 * (therm + nmeas) as f64);
        let manifeste = serde_json::json!({
            "schema": 1, "engine": "kvu-rapide32", "l": l, "beta": p.beta, "beta_nominal": chaines[c].1,
            "seed": p.graine.0, "start": if p.depart == Depart::Chaud { "Hot" } else { "Cold" },
            "therm_steps": therm, "measurements": nmeas, "skip_steps": 1, "or_per_step": nor, "dump_every": dump_every,
            "precision": "f32", "backend": app.backend().nom(), "arch": app.arch(),
            "lot": nch, "plafond": compteurs[c].plafond, "essais_par_lien_hb": essais_par_lien,
            "wall_lot_s": wall, "wall_therm_s": t_therm,
            "provenance": serde_json::to_value(&prov).expect("provenance"),
            "statut": if plafond_total == 0 { "COMPLETE" } else { "PLAFOND" },
            "format_dump": "q64 : float64 (4, L, L, L, L, 4) = quaternions promus du float, sans renormalisation"
        });
        ecrire_json(&d.join("manifest.json"), &manifeste);
        if plafond_total == 0 {
            std::fs::write(d.join("COMPLETE"), "").expect("COMPLETE");
        }
    }
    println!("lot L = {l} : {nch} chaînes, {therm} + {nmeas} pas en {wall:.1} s, plafond {plafond_total}");
    assert_eq!(plafond_total, 0, "plafond d'essais atteint");
}

/// Pilote de chaînes pilotable par l'environnement (un lot = un L) :
/// KVU_F32_L, KVU_F32_BETAS (étiquettes, ex. « 2.3,2.8,3.5 »), KVU_F32_REPS (3 : c1 chaud, c2 chaud, c3 froid),
/// KVU_F32_SEED0, KVU_F32_THERM, KVU_F32_NMEAS, KVU_F32_NOR, KVU_F32_DUMP_EVERY, KVU_F32_FACTEUR_BETA
/// (1,002 pour le contrôle négatif), KVU_F32_SOUS (sous-répertoire de $KVU_LOI_OUT).
#[test]
#[ignore = "GPU"]
fn pilote_chaines() {
    if std::env::var("KVU_F32_L").is_err() {
        println!("SKIP PILOTE-CHAINES (KVU_F32_L absent : test piloté par gates_f32.sh)");
        return;
    }
    let Some(app) = appareil("PILOTE-CHAINES") else { return };
    let Some(racine) = sortie("PILOTE-CHAINES", &env_ou("KVU_F32_SOUS", "chaines/f32")) else { return };
    let l = env_u64("KVU_F32_L", 6) as usize;
    let facteur: f64 = env_ou("KVU_F32_FACTEUR_BETA", "1").parse().expect("facteur");
    let seed0 = env_u64("KVU_F32_SEED0", 20_261_004_000_000);
    let reps = env_u64("KVU_F32_REPS", 3) as usize;
    let mut chaines = Vec::new();
    for (ib, b) in env_ou("KVU_F32_BETAS", "2.3").split(',').enumerate() {
        let beta: f64 = b.trim().parse().expect("β");
        for r in 0..reps {
            let depart = if r + 1 == 3 { Depart::Froid } else { Depart::Chaud };
            let graine = seed0 + 1000 * l as u64 + 10 * ib as u64 + r as u64 + 1;
            chaines.push((format!("L{l}_b{}_c{}", b.trim(), r + 1), b.trim().to_string(), chaine(beta * facteur, graine, depart)));
        }
    }
    ecrire_lot(
        &app,
        l,
        env_u64("KVU_F32_NOR", 4) as usize,
        env_u64("KVU_F32_THERM", 2000) as usize,
        env_u64("KVU_F32_NMEAS", 20_000) as usize,
        env_u64("KVU_F32_DUMP_EVERY", 1000) as usize,
        &racine,
        &chaines,
    );
}

/// Banc F32 (liens/s, non gaté) : bain thermique et overrelaxation, L = 16 et 32 (1 chaîne) et un lot de
/// 16 chaînes à L = 8. Écrit bench_f32.jsonl.
#[test]
#[ignore = "GPU"]
fn bench_f32() {
    let Some(app) = appareil("BENCH-F32") else { return };
    let mut lignes = String::new();
    for (l, nch, nsweeps) in [(16usize, 1usize, 200usize), (32, 1, 60), (8, 16, 400)] {
        let chaines: Vec<ParamChaine> = (0..nch).map(|i| chaine(2.3, 5000 + i as u64, Depart::Chaud)).collect();
        let mut n = nouveau(&app, l, chaines);
        n.init().unwrap();
        for _ in 0..3 {
            pas(&mut n);
        }
        n.synchroniser().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..nsweeps {
            n.sweep_hb().unwrap();
        }
        n.synchroniser().unwrap();
        let t_hb = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        for _ in 0..nsweeps {
            n.sweep_or().unwrap();
        }
        n.synchroniser().unwrap();
        let t_or = t1.elapsed().as_secs_f64();
        let liens = (4 * l.pow(4) * nch * nsweeps) as f64;
        let l_json = serde_json::json!({
            "precision": "f32", "backend": app.backend().nom(), "l": l, "chaines": nch, "nsweeps": nsweeps,
            "hb_liens_par_s": liens / t_hb, "or_liens_par_s": liens / t_or, "pas_par_s": nsweeps as f64 / (t_hb + 4.0 * t_or)
        });
        println!("BENCH {l_json}");
        writeln!(lignes, "{l_json}").expect("mem");
    }
    if let Some(d) = sortie("BENCH-F32", ".") {
        std::fs::write(d.join("bench_f32.jsonl"), lignes).expect("bench");
    }
    verdict("BENCH-F32 (non gaté)", true);
}
