//! Mesure en ligne `bloc:3` de `su2 gen` : à chaque mesure, la configuration grossière (bloc de Bałaban L_b = 3, SU(2)) est calculée
//! sur le GPU (ou sur le CPU si le run est CPU), écrite en q64 et ses observables sont ajoutées à des CSV. Aucun dump fin n'est
//! nécessaire (ARCHITECTURE §4.4, politique de stockage §7). Propriétaire : WP7.
//!
//! Grammaire : `bloc:3:<période>` (période en pas de production, 0 = jamais ; 3 est le pas de bloc, seul accepté).
//! Sorties, dans le répertoire de la chaîne (`MesureEnLigne::mesurer`, argument `sortie`) :
//! - `bloc3_cfg_<pas sur 8 chiffres>.npy` : configuration grossière (L/3)⁴, format q64, hachée à l'écriture ;
//! - `bloc3.csv` : `pas,l_grossier,n_boucles,fraction_hors,theta_max,n_ambigues,plaq_grossiere,sha256_cfg` ;
//! - `bloc3_hist.csv` : `pas,h0,…,h63` (histogramme de θ, 64 classes sur [0, π]).
//!
//! Champ résident F32 (`--precision f32`, WP10) : promu en f64 puis renormalisé en F64 avant le bloc (`promouvoir_renormaliser`).
//!
//! La définition est celle de `rg::bloc` (CPU) : l'écart GPU/CPU est mesuré par la gate RG-GPU (≤ 1e-12), et la mesure en ligne contre
//! `rg bloc` hors ligne sur les dumps d'un même run par ENLIGNE-1. Un run sans `--mesures bloc:3` n'est pas modifié (G-BIT-HIP-PROD).

use super::bloc::{bloquer, OptionsBloc, SortieBloc};
use super::domaine;
use super::geometrie::{Geometrie, PAS};
use super::gpu::BlocGpu;
use super::swendsen::plaquette_moyenne;
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::gpu::Appareil;
use crate::heritage::su2_hb::qnorm;
use crate::obs::enligne::Ecrivain;
use crate::regen::manifeste::{maintenant_iso, parcourir, sha_fichier, SCHEMA_REGEN};
use crate::stockage::npy::{ecrire_champ_hache, FormatChamp};
use crate::stockage::GardeSsd;
use crate::su2::run::MesureEnLigne;
use crate::su2::champ::valeurs_par_chaine;
use crate::su2::{ChampGpu, TamponChamp};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub const NOM_MESURE: &str = "bloc";
pub const NOM_MANIFESTE_REGEN: &str = "regen_bloc3.json";
pub const ENTETE_CSV: &str = "pas,l_grossier,n_boucles,fraction_hors,theta_max,n_ambigues,plaq_grossiere,sha256_cfg";

/// Fils du calcul CPU d'une mesure (le résultat n'en dépend pas : RG-FILS).
fn fils_cpu() -> usize {
    std::env::var("RAYON_NUM_THREADS").ok().and_then(|t| t.parse().ok()).filter(|&n| n >= 1).unwrap_or(4)
}

/// Analyse l'élément `bloc:<pas>:<période>` ; rend la période.
pub fn analyser_element(champs: &[&str], element: &str) -> Resultat<usize> {
    let erreur = |raison: &str| {
        Erreur::Config(format!("--mesures : élément « {element} » invalide ({raison}). Forme attendue : bloc:3:<période>"))
    };
    if champs.len() != 3 {
        return Err(erreur("trois champs attendus"));
    }
    match champs[1].parse::<usize>() {
        Ok(p) if p == PAS => {}
        _ => return Err(erreur(&format!("pas de bloc « {} » : seul 3 est géré", champs[1]))),
    }
    champs[2].parse::<usize>().map_err(|_| erreur(&format!("période « {} » : entier ≥ 0 attendu", champs[2])))
}

/// Ligne de `bloc3.csv`.
pub fn ligne_csv(pas: u64, sortie: &SortieBloc, plaquette: f64, sha: &str) -> String {
    let s = &sortie.stats;
    format!(
        "{pas},{},{},{:.17e},{:.17e},{},{:.17e},{sha}\n",
        sortie.grossier.l,
        s.n,
        s.fraction_hors(),
        s.theta_max,
        s.n_ambigues,
        plaquette
    )
}

/// Ligne de `bloc3_hist.csv`.
pub fn ligne_hist(pas: u64, sortie: &SortieBloc) -> String {
    let h: Vec<String> = sortie.stats.histogramme.iter().map(|v| v.to_string()).collect();
    format!("{pas},{}\n", h.join(","))
}

pub fn entete_hist() -> String {
    let h: Vec<String> = (0..super::bloc::N_CLASSES).map(|i| format!("h{i}")).collect();
    format!("pas,{}", h.join(","))
}

/// Nom du fichier de la configuration grossière au pas `pas`.
pub fn nom_cfg(pas: u64) -> String {
    format!("bloc3_cfg_{pas:08}.npy")
}

/// Liens traités par tâche rayon dans la promotion F32 → F64.
const LIENS_PAR_TACHE: usize = 1 << 14;

/// Champ F32 → champ F64 du bloc (WP10) : chaque valeur est promue en f64 (exact), puis chaque quaternion est renormalisé EN F64
/// (division par la norme, comme `stockage::npy::renormaliser` qui sert aux dumps `cfg_*.npy` d'un run F32). Le bloc de Bałaban
/// est défini en F64 sur des liens SU(2) : sans renormalisation, les liens grossiers `U₀·exp(…)` hériteraient de la norme
/// 1 ± 3·10⁻⁷ du chemin F32 et ne seraient plus unitaires (G-H13). Une passe parallèle, mêmes opérations flottantes que le chemin
/// « f32_vers_f64 puis renormaliser » (test : égalité binaire). Rend le champ et le plus grand défaut ||q| − 1| avant correction.
pub fn promouvoir_renormaliser(src: &[f32], l: usize) -> Resultat<(ChampHote, f64)> {
    if !src.len().is_multiple_of(4) {
        return Err(Erreur::Config(format!("champ F32 : {} valeurs, multiple de 4 requis", src.len())));
    }
    let mut q = vec![0f64; src.len()];
    let resultats: Vec<std::result::Result<f64, usize>> = q
        .par_chunks_mut(4 * LIENS_PAR_TACHE)
        .zip(src.par_chunks(4 * LIENS_PAR_TACHE))
        .enumerate()
        .map(|(b, (dst, s))| {
            let mut pire = 0.0f64;
            for (j, (d, a)) in dst.chunks_exact_mut(4).zip(s.chunks_exact(4)).enumerate() {
                let v = [f64::from(a[0]), f64::from(a[1]), f64::from(a[2]), f64::from(a[3])];
                let n = qnorm(v);
                if !n.is_finite() || n < 1e-300 {
                    return Err(b * LIENS_PAR_TACHE + j);
                }
                pire = pire.max((n - 1.0).abs());
                for (dk, vk) in d.iter_mut().zip(v) {
                    *dk = vk / n;
                }
            }
            Ok(pire)
        })
        .collect();
    let mut pire = 0.0f64;
    for r in resultats {
        match r {
            Ok(p) => pire = pire.max(p),
            Err(i) => return Err(Erreur::Validation(format!("lien {i} nul ou non fini : renormalisation impossible"))),
        }
    }
    Ok((ChampHote { l, q }, pire))
}

pub struct MesureBloc {
    periode: usize,
    garde: Option<GardeSsd>,
    gpu: Option<BlocGpu>,
}

impl MesureBloc {
    pub fn nouvelle(periode: usize, garde: Option<GardeSsd>) -> Self {
        Self { periode, garde, gpu: None }
    }

    fn calculer(&mut self, app: Option<&Appareil>, champ: Option<&ChampGpu>, hote: Option<&ChampHote>, chaine: usize) -> Resultat<SortieBloc> {
        match (champ, hote) {
            (Some(c), _) => {
                let app = app.ok_or_else(|| Erreur::Config("champ GPU fourni sans appareil".into()))?;
                if self.gpu.is_none() {
                    self.gpu = Some(BlocGpu::nouveau(app, Geometrie::TroisCentre)?);
                }
                let g = self.gpu.as_ref().ok_or_else(|| Erreur::Config("module GPU du bloc absent".into()))?;
                match &c.tampon {
                    TamponChamp::F64(_) => g.bloquer_resident(c, chaine, true),
                    // WP10 : champ résident F32 (génération `--precision f32`). La définition du bloc est F64 : le champ de la chaîne est
                    // rapatrié, promu en f64 et renormalisé en F64 (comme ses dumps), puis bloqué par le noyau F64 ; coût d'une mesure :
                    // une lecture F32, une passe parallèle, un téléversement F64.
                    TamponChamp::F32(t) => {
                        let n = valeurs_par_chaine(c.l);
                        if chaine >= c.nchaines || c.nchaines.checked_mul(n) != Some(t.len()) {
                            return Err(Erreur::Config("chaîne ou longueur du champ résident F32 invalide".into()));
                        }
                        let mut h32 = vec![0f32; n];
                        app.lire(t, chaine * n, &mut h32)?;
                        let (promu, _) = promouvoir_renormaliser(&h32, c.l)?;
                        g.bloquer_hote(&promu, true)
                    }
                }
            }
            (None, Some(h)) => bloquer(h, &OptionsBloc { geometrie: Geometrie::TroisCentre, fils: fils_cpu(), angle_so3: false }),
            (None, None) => Err(Erreur::Config("bloc:3 : ni champ GPU ni champ hôte".into())),
        }
    }
}

impl MesureEnLigne for MesureBloc {
    fn nom(&self) -> &str {
        NOM_MESURE
    }
    fn periode(&self) -> usize {
        self.periode
    }
    fn mesurer(&mut self, app: Option<&Appareil>, champ: Option<&ChampGpu>, hote: Option<&ChampHote>, chaine: usize, pas: u64, sortie: &Path) -> Resultat<()> {
        let s = self.calculer(app, champ, hote, chaine)?;
        let plaquette = plaquette_moyenne(&s.grossier);
        let ecrivain = Ecrivain::nouveau(self.garde.clone());
        let garde = self.garde.clone().unwrap_or_else(|| GardeSsd::nouvelle(0.0));
        std::fs::create_dir_all(sortie).map_err(|e| Erreur::io(format!("création de {}", sortie.display()), e))?;
        let ecrit = ecrire_champ_hache(&sortie.join(nom_cfg(pas)), &s.grossier, FormatChamp::Q64, &garde)?;
        ecrivain.ajouter(&sortie.join("bloc3.csv"), ENTETE_CSV, &ligne_csv(pas, &s, plaquette, ecrit.sha256.as_deref().unwrap_or("")))?;
        ecrivain.ajouter(&sortie.join("bloc3_hist.csv"), &entete_hist(), &ligne_hist(pas, &s))
    }
}

// ------------------------------------------------------------------------------------------
// Manifeste kvu-regen/1 d'un run avec mesures en ligne
// ------------------------------------------------------------------------------------------

fn nom_options(manifeste: &Value) -> &'static str {
    let off = manifeste.pointer("/options_numeriques/fp_contract").and_then(Value::as_str) == Some("off");
    if manifeste.get("precision").and_then(Value::as_str) == Some("f32") {
        "rapide"
    } else if off {
        "ref-k1"
    } else {
        "compat-prod"
    }
}

fn sha_ou_null(v: Option<&Value>) -> Value {
    match v.and_then(Value::as_str) {
        Some(s) if s.len() == 64 => json!(s),
        _ => Value::Null,
    }
}

/// Construit le manifeste `kvu-regen/1` d'un run natif (`manifest.json`, schéma `kevinotron-unifie/1`) qui porte des mesures en ligne.
/// `fichiers` : (nom relatif, sha256, octets) de tous les fichiers du run (hors manifestes). `parametres.mesures` = la spécification.
/// Fonction pure (testée sans GPU). Le classement A/B reste à WP4 : l'origine natale n'est pas reconnue par `regen::classer` ⇒ classe C.
pub fn manifeste_regen(natif: &Value, sha_natif: &str, chemin_natif: &str, spec: &str, fichiers: &[(String, String, u64)]) -> Resultat<Value> {
    let manque = |c: &str| Erreur::Format(format!("manifest.json natif : champ « {c} » absent ou invalide"));
    let cfg = natif.get("config_run").ok_or_else(|| manque("config_run"))?;
    let lot = cfg.get("lot").ok_or_else(|| manque("config_run.lot"))?;
    let chaine0 = lot.pointer("/chaines/0").ok_or_else(|| manque("config_run.lot.chaines[0]"))?;
    let u = |p: &str| cfg.pointer(p).and_then(Value::as_u64).ok_or_else(|| manque(p));
    let depart = match chaine0.get("depart").and_then(Value::as_str) {
        Some("froid") => "cold",
        _ => "hot",
    };
    let argv: Vec<String> = natif
        .get("argv")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
        .unwrap_or_default();
    let commande: Vec<String> = match argv.iter().position(|a| a == "su2") {
        Some(i) => argv[i..].to_vec(),
        None => vec!["su2".into(), "gen".into()],
    };
    let modules: Vec<Value> = natif
        .get("modules_compiles")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|m| {
                    let args: Vec<String> = m
                        .get("arguments")
                        .and_then(Value::as_array)
                        .map(|x| x.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
                        .unwrap_or_default();
                    json!({
                        "source_sha256": sha_ou_null(m.get("sha256_source")).as_str().unwrap_or(&"0".repeat(64)),
                        "options": args.join(" "),
                        "arguments": args,
                        "objet_sha256": sha_ou_null(m.get("sha256_code")),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let backend = natif.get("backend").and_then(Value::as_str).filter(|b| ["hip", "cuda", "cpu"].contains(b)).ok_or_else(|| manque("backend"))?;
    let arch = natif.get("arch").cloned().unwrap_or(Value::Null);
    let nom_machine = natif.get("machine").and_then(Value::as_str).unwrap_or("inconnue");
    let graine = chaine0.get("graine").and_then(Value::as_u64);
    let betas = natif.get("betas").cloned().unwrap_or(Value::Null);
    let statut = if natif.get("statut").and_then(Value::as_str) == Some("termine") { "COMPLETE" } else { "INCOMPLETE" };
    let fichiers_json: Vec<Value> = fichiers.iter().map(|(n, s, o)| json!({"nom": n, "sha256": s, "octets": o})).collect();
    Ok(json!({
        "schema": SCHEMA_REGEN,
        "origine": {
            "schema": "kevinotron-unifie/1",
            "manifeste": chemin_natif,
            "sha256_manifeste": sha_natif,
            "adaptateur": "kvu-rg-en-ligne/1",
        },
        "binaire_sha256": sha_ou_null(natif.get("binaire_sha256")),
        "binaire_provenance": "manifeste",
        "git_head": natif.get("git_head").and_then(Value::as_str).unwrap_or("non_enregistre"),
        "source_hash": sha_ou_null(natif.get("source_hash")),
        "noyaux_embarques": natif.get("noyaux_embarques").cloned().unwrap_or_else(|| json!([])),
        "modules_compiles": modules,
        "backend": backend,
        "arch": arch,
        "versions": natif.get("versions").cloned().unwrap_or_else(|| json!({})),
        "machine": {"nom": nom_machine, "cpu": Value::Null, "gpu": natif.get("gpu").cloned().unwrap_or(Value::Null), "ram": Value::Null},
        "graines": {"cle": graine, "etiquettes": [natif.get("rng").and_then(Value::as_str).unwrap_or("")]},
        "parametres": {
            "groupe": "SU(2)",
            "L": lot.get("l").and_then(Value::as_u64).ok_or_else(|| manque("config_run.lot.l"))?,
            "beta": chaine0.get("beta").and_then(Value::as_f64).ok_or_else(|| manque("config_run.lot.chaines[0].beta"))?,
            "betas": betas,
            "start": depart,
            "therm": u("/therm")?,
            "nmeas": u("/nmeas")?,
            "skip": u("/skip")?,
            "nor": lot.get("nor").and_then(Value::as_u64).ok_or_else(|| manque("config_run.lot.nor"))?,
            "precision": natif.get("precision").and_then(Value::as_str).unwrap_or("f64"),
            "options": nom_options(natif),
            "dump_every": u("/dump_every")?,
            "bord": false,
            "mesures": spec,
        },
        "commande": commande,
        "entrees": [],
        "fichiers": fichiers_json,
        "fichiers_hachage": "disque",
        "duree_s": natif.get("wall_s").cloned().unwrap_or(Value::Null),
        "statut": statut,
        "classe_regen": "C",
        "raisons_classe": ["manifeste natif kevinotron-unifie/1 : le classement A/B (regen::classer) ne reconnaît que les adaptateurs su2_hip/1 et su2_gen/1 ; à étendre par WP4 avant tout rejeu REGEN-2"],
        "regen_verifiee": {"date": Value::Null, "machine": Value::Null, "resultat": "non_verifiee"},
        "ecrit_le": maintenant_iso(),
    }))
}

/// Fichiers transitoires d'un run (points de reprise) : ce ne sont pas des sorties, ils ne figurent pas au manifeste (WP10 : un champ de
/// reprise de L = 54 pèse 1,09 Go ; le hacher et le lister serait faux dès que l'orchestrateur le retire).
pub fn est_transitoire(nom: &str) -> bool {
    let base = nom.rsplit('/').next().unwrap_or(nom);
    base == "reprise.json" || base == "reprise.champs" || base.starts_with("reprise.champs.")
}

/// Écrit `<outdir>/regen_bloc3.json` (kvu-regen/1, validé par le schéma embarqué) pour un run terminé qui portait `bloc:3`.
/// Les fichiers listés sont tous ceux du run (hors `manifest.json`, `COMPLETE` et ce manifeste), hachés sur disque.
pub fn ecrire_manifeste_regen(outdir: &Path, spec: &str) -> Resultat<PathBuf> {
    let chemin_natif = outdir.join(crate::heritage::manifest::NOM_MANIFESTE);
    let octets = std::fs::read(&chemin_natif).map_err(|e| Erreur::io(format!("lecture de {}", chemin_natif.display()), e))?;
    let natif: Value = serde_json::from_slice(&octets).map_err(|e| Erreur::Format(format!("{} : {e}", chemin_natif.display())))?;
    let sha_natif = crate::regen::manifeste::sha_octets(&octets);
    let mut fichiers = Vec::new();
    for f in parcourir(outdir)? {
        let nom = f.relatif.clone();
        if nom == crate::heritage::manifest::NOM_MANIFESTE || nom == crate::heritage::manifest::NOM_COMPLETE || nom == NOM_MANIFESTE_REGEN || nom.ends_with(".part") || est_transitoire(&nom) {
            continue;
        }
        let sha = sha_fichier(&outdir.join(&nom))?;
        fichiers.push((nom, sha, f.octets));
    }
    let doc = manifeste_regen(&natif, &sha_natif, &chemin_natif.to_string_lossy(), spec, &fichiers)?;
    let erreurs = crate::regen::schema::valider_manifeste(&doc);
    if !erreurs.is_empty() {
        return Err(Erreur::Validation(format!("manifeste kvu-regen/1 invalide : {}", erreurs.join(" ; "))));
    }
    let chemin = outdir.join(NOM_MANIFESTE_REGEN);
    domaine::ecrire_json(&chemin, &doc)?;
    Ok(chemin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grammaire() {
        assert_eq!(analyser_element(&["bloc", "3", "10"], "bloc:3:10").unwrap(), 10);
        assert_eq!(analyser_element(&["bloc", "3", "0"], "bloc:3:0").unwrap(), 0);
        for mauvais in [vec!["bloc", "2", "10"], vec!["bloc", "3"], vec!["bloc", "3", "x"], vec!["bloc", "3", "1", "2"], vec!["bloc", "", "1"]] {
            assert!(analyser_element(&mauvais, "x").is_err(), "{mauvais:?}");
        }
    }

    /// Générateur congruentiel déterministe (pas de dépendance).
    fn lcg(graine: &mut u64) -> f64 {
        *graine = graine.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*graine >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
    }

    #[test]
    fn promotion_egale_le_chemin_des_dumps() {
        // 3 · 16384 + 5 liens : plusieurs tâches rayon et une queue ; normes ≠ 1 (1 ± 1e-6) comme un champ F32 réel
        let nliens = 3 * LIENS_PAR_TACHE + 5;
        let mut g = 12345u64;
        let src: Vec<f32> = (0..4 * nliens)
            .map(|i| {
                let echelle = if i % 4 == 0 { 1.0 + 2e-6 * lcg(&mut g) } else { 1.0 };
                (lcg(&mut g) * 2.0 * echelle) as f32
            })
            .collect();
        let (promu, pire) = promouvoir_renormaliser(&src, 4).unwrap();
        let mut ref_h = ChampHote { l: 4, q: crate::su2::champ::f32_vers_f64(&src) };
        let pire_ref = crate::stockage::npy::renormaliser(&mut ref_h).unwrap();
        assert_eq!(promu.q.len(), ref_h.q.len());
        assert!(promu.q.iter().zip(&ref_h.q).all(|(a, b)| a.to_bits() == b.to_bits()), "chemin parallèle ≠ chemin des dumps");
        assert_eq!(pire.to_bits(), pire_ref.to_bits());
        // résultat unitaire à la précision machine
        assert!(promu.q.chunks_exact(4).all(|q| (q.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() < 1e-15));
    }

    #[test]
    fn promotion_refuse_un_lien_nul_ou_non_fini() {
        assert!(promouvoir_renormaliser(&[0.0; 8], 2).is_err());
        assert!(promouvoir_renormaliser(&[1.0, 0.0, 0.0, 0.0, f32::NAN, 0.0, 0.0, 0.0], 2).is_err());
        assert!(promouvoir_renormaliser(&[1.0; 6], 2).is_err());
        let (c, p) = promouvoir_renormaliser(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(c.q, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(p, 0.0);
    }

    #[test]
    fn fichiers_transitoires_hors_manifeste() {
        for t in ["reprise.json", "reprise.champs", "reprise.champs.new", "reprise.champs.prev", "c001/reprise.champs"] {
            assert!(est_transitoire(t), "{t}");
        }
        for s in ["bloc3.csv", "bloc3_cfg_00000010.npy", "measurements.csv", "reprise_notes.txt", "xreprise.json"] {
            assert!(!est_transitoire(s), "{s}");
        }
    }

    #[test]
    fn noms_et_lignes() {
        assert_eq!(nom_cfg(7), "bloc3_cfg_00000007.npy");
        assert_eq!(entete_hist().split(',').count(), 65);
        assert_eq!(ENTETE_CSV.split(',').count(), 8);
    }
}
