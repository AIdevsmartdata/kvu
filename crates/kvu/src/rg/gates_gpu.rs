//! Gates GPU du bloc de Bałaban (GATES v1 §4, v3 §v3.3) : RG-GPU, RG-DOMAINE-GPU, G-DET (+ mutant RG-GPU-MUT) sur un backend GPU, et
//! RG-HIP-CUDA (comparaison de deux dossiers de sorties produits par deux backends). Propriétaire : WP7.
//!
//! Seuils gelés (GATES v3 §v3.3), écrits avant tout run :
//! - **RG-GPU** : max sur les liens grossiers de max |Δq| (4 composantes) ≤ 1e-12, GPU contre CPU (`rg::bloc::bloquer`, WP6), sur les 3 fixtures
//!   (`validation/rg/fixtures`) et sur `L24_b8_c1/cfg_002000.npy` (grandL, lecture seule) ; HIP et CUDA ;
//! - **RG-DOMAINE-GPU** : mêmes rapports que le CPU : |Δf| ≤ 1e-12, |Δθ max| ≤ 1e-12, même nombre de branches ambiguës, même nombre de boucles ;
//!   entrées : les 4 de RG-GPU, plus Haar L = 12 (graine de WP6) et les deux grandL L = 12 rapportés (β = 2,3 et 16) quand `--grandl` est donné ;
//!   l'écart maximal d'une classe de l'histogramme est rapporté (INFO) : une classe peut changer d'un compte si θ est à 1 ulp d'une borne ;
//! - **G-DET** : deux exécutions du même backend sur la même entrée donnent les mêmes octets (quaternions et statistiques), dont une avec un module
//!   recompilé (fraîchement chargé) ; entrées : fixture graine 7 et L24 ;
//! - **RG-GPU-MUT** (mutant de l'instrument) : le GPU configuré en géométrie `3coin` contre le CPU `3centre` doit s'écarter de ≥ 1e-3 (sinon ECHEC_OUTIL) ;
//! - **RG-HIP-CUDA** : mêmes entrées, sorties HIP contre CUDA ≤ 1e-12 (+ rapports de domaine comme ci-dessus).

use super::bloc::{bloquer, OptionsBloc, SortieBloc, StatsTheta};
use super::domaine;
use super::gates::{champ_haar, ecart_champs, GRAINES_FIXTURES, GRAINE_HAAR_L12};
use super::geometrie::Geometrie;
use super::gpu::BlocGpu;
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::gpu::Appareil;
use crate::groupes::gates::{LigneGate, Verdict};
use crate::stockage::npy::{ecrire_champ_hache, lire_champ, FormatChamp};
use crate::stockage::GardeSsd;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const SEUIL_GPU: f64 = 1e-12;
pub const SEUIL_DOMAINE_GPU: f64 = 1e-12;
pub const SEUIL_MUTANT_GPU: f64 = 1e-3;
/// Facteur d'extrapolation du temps de L = 24 à L = 54 : (54/24)⁴.
pub const FACTEUR_L54: f64 = 25.628_906_25;
/// Entrée grandL de RG-GPU (lecture seule) : sous-dossier et fichier de `…/ensembles/production`.
pub const GRANDL_L24: (&str, &str) = ("L24_b8_c1", "cfg_002000.npy");

pub struct ContexteGpu<'a> {
    pub app: &'a Appareil,
    pub fixtures: PathBuf,
    pub grandl: Option<PathBuf>,
    /// Dossier où écrire, par entrée, la configuration grossière et ses statistiques (pour RG-HIP-CUDA).
    pub sortie: Option<PathBuf>,
    /// Fils de la référence CPU et du banc CPU.
    pub fils: usize,
    pub mutants: bool,
    /// Répétitions du banc de temps (meilleur temps retenu).
    pub repetitions: usize,
}

fn ligne(ok: bool, id: &str, detail: impl Into<String>) -> LigneGate {
    LigneGate::nouvelle(if ok { Verdict::Pass } else { Verdict::Fail }, id, detail)
}

fn skip(id: &str, detail: impl Into<String>) -> LigneGate {
    LigneGate::nouvelle(Verdict::Skip, id, detail)
}

fn info(id: &str, detail: impl Into<String>) -> LigneGate {
    LigneGate::nouvelle(Verdict::Info, id, detail)
}

/// Une entrée de test : nom, champ, utilisée par RG-GPU (sinon seulement par RG-DOMAINE-GPU).
pub struct Entree {
    pub nom: String,
    pub champ: ChampHote,
    pub pour_rg_gpu: bool,
}

/// Les entrées disponibles : (liste, entrées manquantes obligatoires).
pub fn entrees(ctx: &ContexteGpu) -> Resultat<(Vec<Entree>, Vec<String>)> {
    let mut v = Vec::new();
    let mut manquantes = Vec::new();
    for g in GRAINES_FIXTURES {
        let p = ctx.fixtures.join(format!("cfg_s{g}_q64.npy"));
        v.push(Entree { nom: format!("fixture_s{g}"), champ: lire_champ(&p)?, pour_rg_gpu: true });
    }
    match &ctx.grandl {
        Some(racine) => {
            let p = racine.join(GRANDL_L24.0).join(GRANDL_L24.1);
            if p.exists() {
                v.push(Entree { nom: GRANDL_L24.0.to_string(), champ: lire_champ(&p)?, pour_rg_gpu: true });
            } else {
                manquantes.push(format!("{} absent", p.display()));
            }
            for (nom, dossier, fichier) in super::gates::GRANDL {
                let p = racine.join(dossier).join(fichier);
                if p.exists() {
                    v.push(Entree { nom: nom.to_string(), champ: lire_champ(&p)?, pour_rg_gpu: false });
                }
            }
        }
        None => manquantes.push("--grandl absent (L24_b8_c1)".to_string()),
    }
    v.push(Entree { nom: "haar_L12".to_string(), champ: champ_haar(12, GRAINE_HAAR_L12), pour_rg_gpu: false });
    Ok((v, manquantes))
}

fn options_cpu(fils: usize) -> OptionsBloc {
    OptionsBloc { geometrie: Geometrie::TroisCentre, fils, angle_so3: false }
}

/// Écarts de statistiques entre deux sorties : (|Δf|, |Δθ max|, Δ n_ambiguës, Δ n, écart max d'une classe).
pub fn ecart_stats(a: &StatsTheta, b: &StatsTheta) -> (f64, f64, u64, u64, u64) {
    let dh = a.histogramme.iter().zip(&b.histogramme).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0);
    (
        (a.fraction_hors() - b.fraction_hors()).abs(),
        (a.theta_max - b.theta_max).abs(),
        a.n_ambigues.abs_diff(b.n_ambigues),
        a.n.abs_diff(b.n),
        dh,
    )
}

fn octets_identiques(a: &SortieBloc, b: &SortieBloc) -> bool {
    a.grossier.l == b.grossier.l
        && a.grossier.q.len() == b.grossier.q.len()
        && a.grossier.q.iter().zip(&b.grossier.q).all(|(x, y)| x.to_bits() == y.to_bits())
        && a.stats == b.stats
        && a.stats.theta_max.to_bits() == b.stats.theta_max.to_bits()
}

/// Écrit la configuration grossière (q64) et ses statistiques pour la comparaison HIP/CUDA.
fn ecrire_sortie(dossier: &Path, nom: &str, s: &SortieBloc) -> Resultat<()> {
    std::fs::create_dir_all(dossier).map_err(|e| Erreur::io(format!("création de {}", dossier.display()), e))?;
    ecrire_champ_hache(&dossier.join(format!("{nom}.npy")), &s.grossier, FormatChamp::Q64, &GardeSsd::machine())?;
    domaine::ecrire_json(&dossier.join(format!("{nom}.theta.json")), &domaine::theta_json(&s.stats))
}

/// Batterie complète sur le backend de `ctx.app`.
pub fn batterie_gpu(ctx: &ContexteGpu) -> Resultat<Vec<LigneGate>> {
    let backend = ctx.app.backend().nom();
    let (entrees, manquantes) = entrees(ctx)?;
    let bloc = BlocGpu::nouveau(ctx.app, Geometrie::TroisCentre)?;
    let mut lignes = Vec::new();
    lignes.push(info(
        "RG-GPU-APPAREIL",
        format!("backend {backend}, {} ({}), module rg_bloc sha256 {}…", ctx.app.nom(), ctx.app.arch(), &bloc.provenance().sha256_code[..16.min(bloc.provenance().sha256_code.len())]),
    ));
    let mut max_gpu = 0.0f64;
    let mut max_mat = 0.0f64;
    let mut details_gpu = Vec::new();
    let mut ok_dom = true;
    let mut details_dom = Vec::new();
    let mut n_rg_gpu = 0;
    for e in &entrees {
        let cpu = bloquer(&e.champ, &options_cpu(ctx.fils))?;
        let gpu = bloc.bloquer_hote(&e.champ, true)?;
        let m = ecart_champs(&cpu.grossier, &gpu.grossier);
        let (df, dt, da, dn, dh) = ecart_stats(&cpu.stats, &gpu.stats);
        if e.pour_rg_gpu {
            max_gpu = max_gpu.max(m.comp);
            max_mat = max_mat.max(m.mat);
            n_rg_gpu += 1;
        }
        details_gpu.push(format!("{} (L {}→{}) {:.2e}{}", e.nom, e.champ.l, gpu.grossier.l, m.comp, if e.pour_rg_gpu { "" } else { " [hors gate]" }));
        let ok = df <= SEUIL_DOMAINE_GPU && dt <= SEUIL_DOMAINE_GPU && da == 0 && dn == 0;
        ok_dom &= ok;
        details_dom.push(format!(
            "{} : f {:.6e} (Δ {:.1e}), θmax {:.6} (Δ {:.1e}), ambiguës {} (Δ {da}), n Δ{dn}, classe max Δ{dh}",
            e.nom,
            gpu.stats.fraction_hors(),
            df,
            gpu.stats.theta_max,
            dt,
            gpu.stats.n_ambigues
        ));
        if let Some(d) = &ctx.sortie {
            ecrire_sortie(&d.join(backend), &e.nom, &gpu)?;
        }
    }
    if manquantes.is_empty() && n_rg_gpu == 4 {
        lignes.push(ligne(
            max_gpu <= SEUIL_GPU,
            "RG-GPU",
            format!("backend {backend} : max |Δq| = {max_gpu:.3e} (seuil {SEUIL_GPU:.0e}), module des entrées 2×2 {max_mat:.3e} ; {}", details_gpu.join(" ; ")),
        ));
        lignes.push(ligne(ok_dom, "RG-DOMAINE-GPU", format!("backend {backend} (seuils {SEUIL_DOMAINE_GPU:.0e} sur f et θ max, ambiguës égales) : {}", details_dom.join(" | "))));
    } else {
        let raison = format!("entrée obligatoire manquante : {}", manquantes.join(", "));
        lignes.push(skip("RG-GPU", raison.clone()));
        lignes.push(skip("RG-DOMAINE-GPU", raison));
    }

    // Mutant de l'instrument : GPU en géométrie 3coin contre CPU 3centre sur la fixture de graine 7.
    if ctx.mutants {
        let f7 = &entrees[0];
        let coin = BlocGpu::nouveau(ctx.app, Geometrie::TroisCoin)?;
        let g = coin.bloquer_hote(&f7.champ, false)?;
        let c = bloquer(&f7.champ, &options_cpu(ctx.fils))?;
        let m = ecart_champs(&c.grossier, &g.grossier).comp;
        lignes.push(ligne(m >= SEUIL_MUTANT_GPU, "RG-GPU-MUT", format!("backend {backend} : mutant 3coin contre CPU 3centre, max |Δq| = {m:.3e} (doit être ≥ {SEUIL_MUTANT_GPU:.0e} : ROUGE voulu)")));
    }

    // G-DET : deux exécutions, dont une avec un module rechargé.
    let mut det_ok = true;
    let mut det_det = Vec::new();
    for e in entrees.iter().filter(|e| e.nom == "fixture_s7" || e.nom == GRANDL_L24.0) {
        let a = bloc.bloquer_hote(&e.champ, true)?;
        let b = bloc.bloquer_hote(&e.champ, true)?;
        let frais = BlocGpu::nouveau(ctx.app, Geometrie::TroisCentre)?;
        let c = frais.bloquer_hote(&e.champ, true)?;
        let ok = octets_identiques(&a, &b) && octets_identiques(&a, &c);
        det_ok &= ok;
        det_det.push(format!("{} : {}", e.nom, if ok { "octets identiques (3 exécutions)" } else { "DIFFÉRENCE" }));
    }
    if det_det.len() == 2 {
        lignes.push(ligne(det_ok, "G-DET", format!("backend {backend} : {}", det_det.join(" ; "))));
    } else {
        lignes.push(skip("G-DET", "fixture graine 7 ou L24 absente"));
    }

    // Banc rapporté (jamais une gate) : L = 24 → 8.
    if let Some(e) = entrees.iter().find(|e| e.nom == GRANDL_L24.0) {
        lignes.push(banc(ctx, &bloc, e)?);
    }
    Ok(lignes)
}

/// Ligne INFO du banc de temps L = 24 → 8 (CPU `ctx.fils` fils, backend) et extrapolation L = 54 → 18.
fn banc(ctx: &ContexteGpu, bloc: &BlocGpu, e: &Entree) -> Resultat<LigneGate> {
    let reps = ctx.repetitions.max(1);
    let meilleur = |f: &mut dyn FnMut() -> Resultat<()>| -> Resultat<f64> {
        let mut m = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            f()?;
            m = m.min(t.elapsed().as_secs_f64());
        }
        Ok(m)
    };
    let t_cpu = meilleur(&mut || bloquer(&e.champ, &options_cpu(ctx.fils)).map(|_| ()))?;
    let t_gpu = meilleur(&mut || bloc.bloquer_hote(&e.champ, false).map(|_| ()))?;
    let t_gpu_stats = meilleur(&mut || bloc.bloquer_hote(&e.champ, true).map(|_| ()))?;
    Ok(info(
        "RG-GPU-TEMPS",
        format!(
            "L {}→{} (meilleur de {reps}) : CPU {} fils {t_cpu:.3} s ; {} sans stats {t_gpu:.3} s, avec stats {t_gpu_stats:.3} s (téléversement et relecture compris) ; \
             extrapolation L 54→18 (×{FACTEUR_L54:.1}) : CPU {:.1} s, {} {:.1} s (avec stats)",
            e.champ.l,
            e.champ.l / 3,
            ctx.fils,
            ctx.app.backend().nom(),
            t_cpu * FACTEUR_L54,
            ctx.app.backend().nom(),
            t_gpu_stats * FACTEUR_L54
        ),
    ))
}

// ------------------------------------------------------------------------------------------
// RG-HIP-CUDA : comparaison de deux dossiers de sorties
// ------------------------------------------------------------------------------------------

fn lire_theta(chemin: &Path) -> Resultat<Value> {
    let t = std::fs::read_to_string(chemin).map_err(|e| Erreur::io(format!("lecture de {}", chemin.display()), e))?;
    serde_json::from_str(&t).map_err(|e| Erreur::Format(format!("{} : {e}", chemin.display())))
}

/// Compare deux dossiers `A` et `B` (sorties de `gates-gpu --sortie` de deux backends) : RG-HIP-CUDA, une ligne.
pub fn comparer_dossiers(a: &Path, b: &Path, min_entrees: usize, id: &str, avec_theta: bool) -> Resultat<Vec<LigneGate>> {
    let mut noms: Vec<String> = std::fs::read_dir(a)
        .map_err(|e| Erreur::io(format!("lecture de {}", a.display()), e))?
        .filter_map(|x| x.ok())
        .filter_map(|x| x.file_name().to_str().and_then(|n| n.strip_suffix(".npy")).map(str::to_string))
        .collect();
    noms.sort();
    let mut ok = true;
    let mut max_q = 0.0f64;
    let mut details = Vec::new();
    let mut n = 0;
    for nom in &noms {
        let (pa, pb) = (b.join(format!("{nom}.npy")), a.join(format!("{nom}.npy")));
        if !pa.exists() {
            continue;
        }
        let (ca, cb) = (lire_champ(&pb)?, lire_champ(&pa)?);
        if ca.l != cb.l {
            ok = false;
            details.push(format!("{nom} : côtés différents ({} / {})", ca.l, cb.l));
            continue;
        }
        let m = ecart_champs(&ca, &cb).comp;
        max_q = max_q.max(m);
        n += 1;
        if !avec_theta {
            ok &= m <= SEUIL_GPU;
            details.push(format!("{nom} {m:.2e}"));
            continue;
        }
        let (ta, tb) = (lire_theta(&a.join(format!("{nom}.theta.json")))?, lire_theta(&b.join(format!("{nom}.theta.json")))?);
        let f = |v: &Value, k: &str| v[k].as_f64().unwrap_or(f64::NAN);
        let u = |v: &Value, k: &str| v[k].as_u64().unwrap_or(u64::MAX);
        let df = (f(&ta, "fraction_hors") - f(&tb, "fraction_hors")).abs();
        let dt = (f(&ta, "theta_max") - f(&tb, "theta_max")).abs();
        let da = u(&ta, "n_ambigues").abs_diff(u(&tb, "n_ambigues"));
        let bon = m <= SEUIL_GPU && df <= SEUIL_DOMAINE_GPU && dt <= SEUIL_DOMAINE_GPU && da == 0 && u(&ta, "n_boucles") == u(&tb, "n_boucles");
        ok &= bon;
        details.push(format!("{nom} {m:.2e} (Δf {df:.1e}, Δθmax {dt:.1e}, Δamb {da})"));
    }
    if n < min_entrees {
        return Ok(vec![LigneGate::nouvelle(Verdict::Fail, id, format!("{n} entrée(s) communes seulement (≥ {min_entrees} requises) entre {} et {}", a.display(), b.display()))]);
    }
    Ok(vec![ligne(
        ok,
        id,
        format!("{} contre {} : max |Δq| = {max_q:.3e} (seuil {SEUIL_GPU:.0e}) sur {n} entrées : {}", a.display(), b.display(), details.join(" ; ")),
    )])
}

/// Résumé JSON d'une batterie, pour les certificats.
pub fn lignes_json(l: &[LigneGate]) -> Vec<Value> {
    l.iter().map(|x| json!({"verdict": x.verdict.to_string(), "id": x.id, "detail": x.detail})).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ecarts_de_statistiques() {
        let mut a = StatsTheta::vide();
        let mut b = StatsTheta::vide();
        a.ajouter(0.1, false, false);
        b.ajouter(0.1, false, false);
        assert_eq!(ecart_stats(&a, &b), (0.0, 0.0, 0, 0, 0));
        b.ajouter(2.0, true, false);
        let (df, dt, da, dn, dh) = ecart_stats(&a, &b);
        assert!(df > 0.4 && dt > 1.8 && da == 1 && dn == 1 && dh == 1);
    }

    #[test]
    fn facteur_d_extrapolation() {
        assert!((FACTEUR_L54 - (54.0f64 / 24.0).powi(4)).abs() < 1e-9);
    }
}
