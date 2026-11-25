//! Chemin C1 sur GPU (B10, kvu v0.2), lots A (torsion à côtés égaux, F64) et C (masque de liens gelés à côtés égaux : géométrie
//! `Bord::Fixe`/boîte, `fixer_masque` WP-DIR, `fixer_gel` R2-3 traduit en masque).
//!
//! Les signatures, grilles et dispositions des noyaux sont décrites dans `noyaux/su2_geo.cu`.
//! Ce fichier est embarqué ici hors du registre gelé `crate::noyaux::TOUS`, comme `su2_boite.cu`.
//! Le module assemble `su2_commun.h`, `su2_ref64.cu` et `su2_geo.cu` : les réductions C0 sont réutilisées.
//! `Ref64Gpu`, `Rapide32Gpu` et `Cpu64` ne sont pas modifiés.
//!
//! CONTRAT CONSOMMÉ : celui de l'en-tête de `geom/tables.rs` (socle 4042d25). Point par point, où il est appliqué et vérifié :
//! - « s = ((x0·ℓ1 + x1)·ℓ2 + x2)·ℓ3 + x3, `pas`, li = μ·V + s, voisins périodiques » : tampon `geo` copié de `Tables` (`tableau_geo`),
//!   aucune formule de pas sur GPU ; porte G-GEO (voisins et classes du GPU = `Tables`, exhaustif).
//! - « c(p, s) = `coef[p·V + s]`, 6V réels, décalage p·V ; 1 partout si `coef` = None ; le β GLOBAL n'est PAS dans `coef` » :
//!   `coef` téléversé tel quel ; `alpha = betas[c] · |A|` dans `k_heatbath_geo` ; portes P2/P2-COMPAT (coef de 1.0 = C0 à l'octet).
//! - « agrafe : plaquette haute de coin s, coefficient `coef[plan(μ,ν)·V + s]` ; basse de coin s − ν̂, coefficient
//!   `coef[plan(μ,ν)·V + voisin_bas(s, ν)]` » : `g_agrafe` de `su2_geo.cu` ; porte P3 et mutants 1 (plan), 2 (pile décalée),
//!   3 (z bas oublié), 5 (z bas lu au coin s).
//! - « masque : 4V octets indexés par li ; 1 = lien gelé » : `recalculer_masques` (union, octet non nul = gelé) ; portes C-MASQUE-*.
//! - « classes = SITES ; balayage `for μ { for classe }`, parité 0 puis 1 à côtés pairs ; côtés impairs : CPU seulement, les
//!   noyaux GPU de B10 refusent » : boucles `mu` puis `p` de `hb`/`sweep_or` ; refus `Coloriage::Parite` du constructeur (et L
//!   pair par `su2::valider`) ; porte G-GEO (ordre des parités).

use super::boite::masque_gel_x0_spatial;
use super::champ::ChampGpu;
use super::torsion::Torsion;
use super::{AppMasque, Compteurs, ConfigLot, Gel, NoyauxSu2};
use crate::commun::{BackendKind, ChampHote, Depart, Erreur, Precision, Resultat};
use crate::geom::{Bord, Chemin, Coloriage, GeoSpec, SourceBord, SourceMasque, Tables};
use crate::gpu::{
    assembler, Appareil, Arg, Fonction, Lancement, Module, OptionsCompil, ProvenanceCompil,
    SourceAssemblee, Tampon,
};
use crate::noyaux::{Fichier, SU2_COMMUN, SU2_REF64};

/// Source C1, conservée hors du registre gelé des noyaux C0.
pub static SU2_GEO: Fichier = Fichier {
    nom: "su2_geo.cu",
    contenu: include_str!("../../noyaux/su2_geo.cu"),
};

/// Direction du temps d'un run `su2 gen` (convention historique : axe 3, colonne `temporal=3` du manifeste).
pub const DIR_TEMPS_SU2_GEN: u8 = 3;

/// Libellé `moteur_su2` du manifeste d'un run du chemin C1.
pub const MOTEUR_SU2_C1: &str = "kvu::su2::geo_gpu::Geo64Gpu (chemin C1 sur GPU, B10)";

/// Géométrie C1 d'un run `su2 gen` tordu sur GPU : L⁴ périodique, torsion, temps = axe 3. Source UNIQUE (moteur, manifeste,
/// `geometrie.json`, contrôles de reprise).
pub fn spec_su2_gen(l: usize, torsion: Torsion) -> GeoSpec {
    GeoSpec::cubique(l, DIR_TEMPS_SU2_GEN).avec_torsion(torsion)
}

/// Moteur consigné par `Cpu64` tordu (M4) dans ses invariants de reprise.
pub const MOTEUR_CPU_TORDU: &str = "Cpu64+M4";

/// Nom du moteur C1 consigné dans les invariants de reprise : « Geo64Gpu/hip » ou « Geo64Gpu/cuda ».
pub fn nom_moteur_c1(backend: BackendKind) -> String {
    format!("Geo64Gpu/{}", backend.nom())
}

/// Invariants de reprise d'un moteur C1, construits SANS GPU : source unique de `Geo64Gpu::invariants_reprise` et du contrôle
/// avant ouverture du GPU (revue c09b, point 2).
pub fn invariants_c1(
    backend: &str,
    spec: &GeoSpec,
    gel: Gel,
    masque_ext: Option<&[u8]>,
    app: AppMasque,
) -> serde_json::Value {
    serde_json::json!({
        "moteur": format!("Geo64Gpu/{backend}"),
        "geometrie": spec.bloc(),
        "gel": serde_json::to_value(gel).unwrap_or(serde_json::Value::Null),
        "masque_externe_sha256": masque_ext.map(crate::heritage::manifest::sha256_hex),
        "app_masque": app.nom(),
    })
}

/// Invariants de reprise de `Cpu64` tordu (M4) : moteur, torsion ; accepte un point de reprise ancien de provenance CPU.
pub fn invariants_cpu_tordu(t: &Torsion) -> serde_json::Value {
    serde_json::json!({ "moteur": MOTEUR_CPU_TORDU, "torsion": t.texte(), "accepte_sans_invariants": true })
}

/// Invariants qu'aura le moteur d'une reprise `su2 gen` de côté `l`, torsion et backend donnés (sans GPU) : None sans torsion
/// (C0) ; `Cpu64+M4` sur CPU ; sinon C1 (`su2 gen` ne pose ni gel, ni masque externe : gel aucun, `AppMasque::Tout`).
pub fn invariants_attendus_su2_gen(
    l: usize,
    torsion: &Torsion,
    backend: &str,
) -> Option<serde_json::Value> {
    if torsion.est_triviale() {
        None
    } else if backend == "cpu" {
        Some(invariants_cpu_tordu(torsion))
    } else {
        Some(invariants_c1(
            backend,
            &spec_su2_gen(l, *torsion),
            Gel::Aucun,
            None,
            AppMasque::Tout,
        ))
    }
}

/// Contrôle PUR (lecture de `reprise.json` et du manifeste, aucun GPU) d'une reprise `su2 gen`, appelé AVANT l'ouverture du GPU
/// et avant toute écriture : les invariants consignés doivent ÉGALER les invariants attendus (moteur, géométrie, gel, masque,
/// `AppMasque`) ; sans invariants (binaire antérieur), seul un run tordu de provenance CPU PROUVÉE au manifeste se reprend, sur
/// CPU. Sans torsion (C0) ou sans `reprise.json`, rien n'est lu (ordre historique des erreurs). Refus = `Erreur::Config` (code 2).
pub fn verifier_reprise_su2_gen(
    dir: &std::path::Path,
    l: usize,
    torsion: &Torsion,
    backend: &str,
) -> Resultat<()> {
    if torsion.est_triviale() || !dir.join(super::reprise::NOM_JSON).exists() {
        return Ok(());
    }
    let etat = super::reprise::lire_etat_seul(dir)?;
    let attendu = invariants_attendus_su2_gen(l, torsion, backend);
    match (&etat.invariants, &attendu) {
        (Some(c), Some(a)) if c == a => Ok(()),
        (Some(c), a) => {
            let ecarts: Vec<String> = match (c.as_object(), a.as_ref().and_then(|a| a.as_object())) {
                (Some(co), Some(ao)) => {
                    let mut k: Vec<&String> = co.keys().chain(ao.keys()).collect();
                    k.sort();
                    k.dedup();
                    k.into_iter().filter(|k| co.get(*k) != ao.get(*k)).cloned().collect()
                }
                _ => vec!["moteur".into()],
            };
            Err(Erreur::Config(format!(
                "reprise de {} : invariants consignés différents des invariants attendus (--backend {backend}) sur [{}] ; \
                 consignés {c}, attendus {} : un run se reprend sur son moteur d'origine, avec sa géométrie, son gel et son masque",
                dir.display(),
                ecarts.join(", "),
                a.as_ref().map_or("aucun (C0)".to_string(), |a| a.to_string())
            )))
        }
        (None, _) if backend == "cpu" && super::reprise::provenance_cpu(dir) == Some(true) => Ok(()),
        (None, _) => Err(Erreur::Config(format!(
            "reprise de {} : point de reprise SANS invariants (binaire antérieur) d'un run tordu ; moteur d'origine (provenance au manifeste) : \
             {} ; seul un run tordu créé sur CPU se reprend, avec --backend cpu ; un run tordu créé sur GPU avant c09b (ou de \
             provenance indéterminée) ne se reprend pas : le relancer avec le binaire actuel",
            dir.display(),
            match super::reprise::provenance_cpu(dir) {
                Some(true) => "CPU",
                Some(false) => "GPU",
                None => "indéterminée (manifeste absent ou illisible)",
            }
        ))),
    }
}

/// Provenance du fichier de noyaux C1 (hors registre gelé `noyaux::TOUS`), pour le manifeste d'un run C1 (`noyaux_c1`) :
/// nom, sha256 et TEXTE intégral (revue c09b, point 4 : l'empreinte seule ne permet pas de relire la source dans les sorties).
pub fn provenance_c1() -> serde_json::Value {
    serde_json::json!([{
        "nom": SU2_GEO.nom,
        "sha256": crate::heritage::manifest::sha256_hex(SU2_GEO.contenu.as_bytes()),
        "texte": SU2_GEO.contenu,
    }])
}

/// Taille de bloc de K1, dont dépend l'arbre de réduction.
const BS: u32 = 256;

/// Source du module C1 : [SU2_COMMUN, SU2_REF64, SU2_GEO], sous le nom logique « su2_geo64 ».
/// Un mutant positif ajoute le define ("KVU_GEO_MUTANT", "<n>") pour le contrôle négatif.
pub fn source_geo64(backend: BackendKind, mutant: u8) -> SourceAssemblee {
    let valeur = mutant.to_string();
    let defines = [("KVU_GEO_MUTANT", valeur.as_str())];
    let defines = if mutant > 0 { &defines[..] } else { &[] };
    assembler(
        backend,
        "su2_geo64",
        &[&SU2_COMMUN, &SU2_REF64, &SU2_GEO],
        defines,
    )
}

/// Tableau géométrique copié des tables : [ℓ0, ℓ1, ℓ2, ℓ3, V, pas0, pas1, pas2, pas3].
pub fn tableau_geo(t: &Tables) -> [i64; 9] {
    [
        t.cotes[0] as i64,
        t.cotes[1] as i64,
        t.cotes[2] as i64,
        t.cotes[3] as i64,
        t.v as i64,
        t.pas[0] as i64,
        t.pas[1] as i64,
        t.pas[2] as i64,
        t.pas[3] as i64,
    ]
}

/// Porte G-GEO : voisins (8·V) et sites par parité (V), avec tableau forcé pour le contrôle négatif.
pub fn tester_geometrie(
    app: &Appareil,
    t: &Tables,
    geo_force: Option<[i64; 9]>,
) -> Resultat<(Vec<u32>, Vec<u32>)> {
    if !t.v.is_multiple_of(2) {
        return Err(Erreur::Config(
            "test de géométrie GPU : volume V pair requis".into(),
        ));
    }
    let src = source_geo64(app.backend(), 0);
    let module = app.compiler(&src, &OptionsCompil::reference_k1())?;
    let f = app.fonction(&module, "k_geo_test")?;
    let mut geo = app.alloc::<i64>(9)?;
    app.ecrire(&mut geo, 0, &geo_force.unwrap_or_else(|| tableau_geo(t)))?;
    let voisins = app.alloc::<u32>(8 * t.v)?;
    let sites = app.alloc::<u32>(t.v)?;
    app.lancer(
        &f,
        &Lancement::lineaire(t.v, BS),
        &[geo.arg(), voisins.arg(), sites.arg()],
    )?;
    app.synchroniser()?;
    let mut voisins_h = vec![0u32; 8 * t.v];
    let mut sites_h = vec![0u32; t.v];
    app.lire(&voisins, 0, &mut voisins_h)?;
    app.lire(&sites, 0, &mut sites_h)?;
    Ok((voisins_h, sites_h))
}

/// Mémoire device d'un lot, hors historiques : champ, site, réduction, Polyakov et coefficients partagés.
/// Une marge de 64 mots par chaîne couvre les paramètres et compteurs, comme dans le pilote C0.
pub fn memoire_necessaire_geo(t: &Tables, nchaines: usize) -> u64 {
    octets_device(
        t.cotes,
        t.v,
        t.coef.as_ref().map_or(0, |c| c.len()),
        nchaines,
    )
}

/// O3 (relecture c09) : besoin mémoire d'un run C1 L⁴ tordu de `nchaines` chaînes, pour l'admission de `su2 gen`.
/// GPU = allocations du moteur (`octets_device` : champ, tampons de mesure, coefficients 6V f64, deux masques 4V, `geo`) ;
/// hôte = SUPPLÉMENT au besoin C0 : pic de construction (tables : 6V coefficients f64 + classes 8V octets ; masques
/// géométrie, union et complet : 3 × 4V octets).
pub fn besoin_memoire_c1(l: usize, nchaines: usize) -> crate::machine::admission::BesoinMemoire {
    let v = l.saturating_pow(4);
    crate::machine::admission::BesoinMemoire {
        hote: (v as u64).saturating_mul(48 + 8 + 12),
        gpu: octets_device([l; 4], v, 6 * v, nchaines),
    }
}

/// Revue c09b, point 3 : admission d'un run `su2 gen` C1 ENTIER. GPU = max(estimation C0, allocations du moteur
/// [`besoin_memoire_c1`]) + historique des mesures (5 réels f64 par case : 40 × chaînes × (max(therm, nmeas) + 1) octets) ;
/// hôte = estimation C0 + pic des tables + rapatriement de l'historique (même taille).
pub fn besoin_run_c1(cfg: &super::run::ConfigRun) -> crate::machine::admission::BesoinMemoire {
    let c0 = super::run::besoin_memoire(cfg, true);
    let n = cfg.lot.chaines.len();
    let c1 = besoin_memoire_c1(cfg.lot.l, n);
    let hist = 40u64
        .saturating_mul(n as u64)
        .saturating_mul(cfg.therm.max(cfg.nmeas).saturating_add(1));
    crate::machine::admission::BesoinMemoire {
        gpu: c0.gpu.max(c1.gpu).saturating_add(hist),
        hote: c0.hote.saturating_add(c1.hote).saturating_add(hist),
    }
}

/// Octets device d'un lot (champ, site, réduction, Polyakov par chaîne ; coefficients `ncoef` f64 et tampon `geo` partagés).
fn octets_device(cotes: [usize; 4], v: usize, ncoef: usize, nchaines: usize) -> u64 {
    let v = v as u64;
    let pol: u64 = cotes.iter().map(|&l| v / l as u64).sum();
    let par_chaine = 16 * v + v + v.div_ceil(u64::from(BS)) + pol + 64;
    // + V mots : les deux tampons de masque (« toujours » et « complet », 4V octets chacun)
    8 * (par_chaine * nchaines as u64 + ncoef as u64 + 9 + v)
}

/// Validation PURE (aucun GPU, aucune allocation de tables) d'une spécification pour le chemin C1 GPU d'un lot de côté
/// `l` : appelée par le constructeur ET par `su2 gen` avant `ouvrir_backend` (relecture c09, O5). Toute combinaison non
/// portée est un refus d'usage (`Erreur::Config`, code 2).
pub fn valider_c1(l: usize, spec: &GeoSpec) -> Resultat<()> {
    if l < 2 || !l.is_multiple_of(2) || l > 180 {
        return Err(Erreur::Config(format!(
            "L = {l} : le chemin C1 GPU exige L pair, 2 ≤ L ≤ 180 (compteur Philox sur 32 bits)"
        )));
    }
    spec.valider()?;
    if spec.chemin() == Chemin::C2 {
        return Err(Erreur::Config(
            "chemin C2 (répliques) sur GPU : non porté en v0.2".into(),
        ));
    }
    if spec.cotes_stockees() != [l; 4] {
        return Err(Erreur::Config(format!(
            "côtés {:?} sur GPU : seuls les côtés égaux à L = {l} sont portés (côtés inégaux : lot B, non porté)",
            spec.cotes_stockees()
        )));
    }
    let gele = spec.masque.is_some() || spec.bords.iter().any(|b| !matches!(b, Bord::Periodique));
    if gele && !spec.torsion.est_triviale() {
        return Err(Erreur::Config(
            "torsion et masque ou bord fixé sur GPU : non porté (aucune référence CPU, Cpu64 refuse la combinaison)".into(),
        ));
    }
    if spec.beta_plans.is_some_and(|b| b.iter().any(|&x| x != 1.0)) {
        return Err(Erreur::Config("β par plan sur GPU : non porté".into()));
    }
    // O4 (relecture c09) : CONTRAT — Geo64Gpu gèle des liens mais n'en POSE aucune valeur. Un bord `Fixe` ou une boîte doivent
    // donc déclarer `SourceBord::ParChaine` : l'appelant pose lui-même les valeurs des liens gelés (`televerser`) avant le
    // premier balayage. Une valeur prescrite (froid, chaud, copie, npy) serait ignorée en silence : refus.
    let prescrite = |v: &SourceBord| !matches!(v, SourceBord::ParChaine);
    let bord_prescrit = spec
        .bords
        .iter()
        .any(|b| matches!(b, Bord::Fixe { valeurs } if prescrite(valeurs)));
    let boite_prescrite =
        matches!(&spec.masque, Some(SourceMasque::Boite { valeurs, .. }) if prescrite(valeurs));
    if bord_prescrit || boite_prescrite {
        return Err(Erreur::Config(
            "valeurs de bord prescrites (froid, chaud, copie, npy) : Geo64Gpu ne les pose pas ; déclarer \
             SourceBord::ParChaine et poser les liens gelés par televerser avant le premier balayage"
                .into(),
        ));
    }
    Ok(())
}

/// Générateur C1 F64 sur GPU, construit depuis sa spécification géométrique immuable.
pub struct Geo64Gpu {
    cfg: ConfigLot,
    spec: GeoSpec,
    cotes: [usize; 4],
    v: usize,
    app: Appareil,
    champ: ChampGpu,
    prov: ProvenanceCompil,
    _module: Module,
    f_init: Fonction,
    f_hb: Fonction,
    f_or: Fonction,
    f_plaq_site: Fonction,
    f_reduce: Fonction,
    f_final: Fonction,
    f_pol: Fonction,
    geo: Tampon<i64>,
    coef: Option<Tampon<f64>>,
    betas: Tampon<f64>,
    k0s: Tampon<u32>,
    k1s: Tampon<u32>,
    hot: Tampon<i32>,
    cap: Tampon<u64>,
    tries: Tampon<u64>,
    site: Tampon<f64>,
    part: Tampon<f64>,
    pol: Tampon<f64>,
    hist: Option<(Tampon<f64>, usize)>,
    sweep_id: u32,
    /// Liens gelés par la géométrie (`Tables.masque` : `Bord::Fixe`, boîte), toujours appliqués.
    masque_geo: Option<Vec<u8>>,
    /// Masque posé par `fixer_masque` (WP-DIR), soumis à `app_masque` comme dans `Ref64Gpu`.
    masque_ext: Option<Vec<u8>>,
    /// Gel R2-3 (`fixer_gel`), traduit en masque (`masque_gel_x0_spatial`, porte G-DIR-MASQUE-GEL), toujours appliqué.
    gel: Gel,
    app_masque: AppMasque,
    /// Tampons device : « toujours » = géométrie ∪ gel ; « complet » = toujours ∪ masque_ext (None : égal à « toujours »).
    m_toujours: Option<Tampon<u8>>,
    m_complet: Option<Tampon<u8>>,
}

impl Geo64Gpu {
    pub fn nouveau(app: &Appareil, cfg: ConfigLot, spec: &GeoSpec) -> Resultat<Geo64Gpu> {
        Self::nouveau_mutant(app, cfg, spec, 0)
    }

    /// Construit le moteur ou un contrôle négatif (mutants 1 à 5, jamais en production).
    #[doc(hidden)]
    pub fn nouveau_mutant(
        app: &Appareil,
        cfg: ConfigLot,
        spec: &GeoSpec,
        mutant: u8,
    ) -> Resultat<Geo64Gpu> {
        super::valider(&cfg)?;
        if mutant > 5 {
            return Err(Erreur::Config(format!(
                "mutant C1 GPU {mutant} : valeur attendue entre 0 et 5"
            )));
        }
        if cfg.precision != Precision::F64 {
            return Err(Erreur::Config(
                "le chemin C1 GPU ne calcule qu'en F64".into(),
            ));
        }
        if cfg.mutant_metro {
            return Err(Erreur::Config(
                "mutant Metropolis non défini sur le chemin C1 GPU".into(),
            ));
        }
        if cfg.chaines.len() > super::ref64::CHAINES_MAX {
            return Err(Erreur::Config(format!(
                "{} chaînes : au plus {} par lot",
                cfg.chaines.len(),
                super::ref64::CHAINES_MAX
            )));
        }
        valider_c1(cfg.l, spec)?;
        let ncoef = if spec.torsion.est_triviale() && spec.beta_plans.is_none() {
            0
        } else {
            6 * spec.v()
        };
        let besoin = octets_device(spec.cotes_stockees(), spec.v(), ncoef, cfg.chaines.len());
        if let Ok(libre) = app.memoire_libre_octets() {
            if besoin > libre / 10 * 9 {
                return Err(Erreur::Ressources(format!(
                    "lot de {} chaînes à L = {} : {:.1} Go de mémoire GPU nécessaires, {:.1} Go libres",
                    cfg.chaines.len(),
                    cfg.l,
                    besoin as f64 / crate::stockage::garde::OCTETS_PAR_GO,
                    libre as f64 / crate::stockage::garde::OCTETS_PAR_GO
                )));
            }
        }
        let t = Tables::compiler(spec)?;
        if t.coloriage != Coloriage::Parite {
            return Err(Erreur::Config(
                "côtés impairs : le GPU exige des côtés pairs".into(),
            ));
        }
        if t.cotes != [cfg.l; 4] {
            return Err(Erreur::Config("côtés inégaux sur GPU : non porté".into()));
        }
        if t.coef
            .as_ref()
            .is_some_and(|c| c.iter().any(|&z| z != 1.0 && z != -1.0))
        {
            return Err(Erreur::Config("β par plan sur GPU : non porté".into()));
        }
        if !spec.torsion.est_triviale() && cfg.chaines.iter().any(|c| c.beta == 0.0) {
            return Err(Erreur::Config(
                "torsion de 't Hooft : β = 0 interdit (bain thermique historique singulier)".into(),
            ));
        }
        let backend = app.backend();
        let src = source_geo64(backend, mutant);
        let module = app.compiler(&src, &cfg.options)?;
        let arch = app.arch();
        let prov = ProvenanceCompil {
            nom: src.nom.clone(),
            backend,
            arch: arch.clone(),
            sha256_source: src.sha256.clone(),
            parties: src.parties.clone(),
            options: cfg.options.clone(),
            arguments: cfg.options.arguments(backend, &arch),
            sha256_code: module.sha256_code.clone(),
        };
        let f = |nom: &str| app.fonction(&module, nom);
        let (l, nch, v) = (cfg.l, cfg.chaines.len(), t.v);
        let nb = v.div_ceil(BS as usize);
        let npol: usize = t.cotes.iter().map(|&cote| v / cote).sum();
        let mut geo = app.alloc::<i64>(9)?;
        app.ecrire(&mut geo, 0, &tableau_geo(&t))?;
        let coef = match &t.coef {
            Some(c) => {
                let mut tampon = app.alloc::<f64>(c.len())?;
                app.ecrire(&mut tampon, 0, c)?;
                Some(tampon)
            }
            None => None,
        };
        let mut betas = app.alloc::<f64>(nch)?;
        let mut k0s = app.alloc::<u32>(nch)?;
        let mut k1s = app.alloc::<u32>(nch)?;
        let mut hot = app.alloc::<i32>(nch)?;
        let beta_h: Vec<f64> = cfg.chaines.iter().map(|c| c.beta).collect();
        let cles: Vec<(u32, u32)> = cfg.chaines.iter().map(|c| c.graine.cle()).collect();
        let hot_h: Vec<i32> = cfg
            .chaines
            .iter()
            .map(|c| i32::from(c.depart == Depart::Chaud))
            .collect();
        app.ecrire(&mut betas, 0, &beta_h)?;
        app.ecrire(&mut k0s, 0, &cles.iter().map(|k| k.0).collect::<Vec<_>>())?;
        app.ecrire(&mut k1s, 0, &cles.iter().map(|k| k.1).collect::<Vec<_>>())?;
        app.ecrire(&mut hot, 0, &hot_h)?;

        let mut g = Geo64Gpu {
            champ: ChampGpu::allouer(app, l, nch, Precision::F64)?,
            f_init: f("k_init_geo")?,
            f_hb: f("k_heatbath_geo")?,
            f_or: f("k_overrelax_geo")?,
            f_plaq_site: f("k_plaq_site_geo")?,
            f_reduce: f("k_reduce")?,
            f_final: f("k_final")?,
            f_pol: f("k_polyakov_geo")?,
            geo,
            coef,
            betas,
            k0s,
            k1s,
            hot,
            cap: app.alloc::<u64>(nch)?,
            tries: app.alloc::<u64>(nch)?,
            site: app.alloc::<f64>(nch * v)?,
            part: app.alloc::<f64>(nch * nb)?,
            pol: app.alloc::<f64>(nch * npol)?,
            hist: None,
            sweep_id: 0,
            app: app.clone(),
            prov,
            _module: module,
            cfg,
            spec: spec.clone(),
            cotes: t.cotes,
            v,
            masque_geo: t.masque.clone(),
            masque_ext: None,
            gel: Gel::Aucun,
            app_masque: AppMasque::Tout,
            m_toujours: None,
            m_complet: None,
        };
        g.recalculer_masques()?;
        Ok(g)
    }

    pub fn spec(&self) -> &GeoSpec {
        &self.spec
    }

    fn nch(&self) -> usize {
        self.cfg.chaines.len()
    }

    /// Recalcule et téléverse les deux masques (union octet par octet ; un octet non nul gèle le lien).
    fn recalculer_masques(&mut self) -> Resultat<()> {
        let union = |a: Option<Vec<u8>>, b: Option<&[u8]>| -> Option<Vec<u8>> {
            match (a, b) {
                (Some(mut a), Some(b)) => {
                    for (x, &y) in a.iter_mut().zip(b) {
                        *x |= y;
                    }
                    Some(a)
                }
                (a, None) => a,
                (None, Some(b)) => Some(b.to_vec()),
            }
        };
        let gel = (self.gel == Gel::BordX0Spatial).then(|| masque_gel_x0_spatial(self.cfg.l));
        let toujours = union(self.masque_geo.clone(), gel.as_deref());
        let complet = self
            .masque_ext
            .as_deref()
            .and_then(|e| union(toujours.clone(), Some(e)));
        let app = self.app.clone();
        let televerser = |m: Option<Vec<u8>>| -> Resultat<Option<Tampon<u8>>> {
            m.map(|m| {
                let mut t = app.alloc::<u8>(m.len())?;
                app.ecrire(&mut t, 0, &m)?;
                Ok(t)
            })
            .transpose()
        };
        self.m_toujours = televerser(toujours)?;
        self.m_complet = televerser(complet)?;
        self.app.synchroniser()
    }

    /// Argument « masque » : le masque complet si l'étape respecte le masque WP-DIR, sinon le seul masque « toujours ».
    fn arg_masque(&self, respecte: bool) -> Arg {
        match (respecte, &self.m_complet, &self.m_toujours) {
            (true, Some(c), _) => c.arg(),
            (_, _, Some(t)) => t.arg(),
            _ => Arg::Nul,
        }
    }

    fn tampon_champ(&self) -> Resultat<&Tampon<f64>> {
        match &self.champ.tampon {
            super::TamponChamp::F64(t) => Ok(t),
            super::TamponChamp::F32(_) => {
                Err(Erreur::Config("champ F32 dans un générateur F64".into()))
            }
        }
    }

    fn lancement_chaines(&self, n: usize) -> Lancement {
        Lancement::lineaire_lot(n, self.nch(), BS)
    }

    /// Un bain thermique et ses essais par lien : 4·V diagnostics par chaîne, comme dans le pilote C0.
    pub fn sweep_hb_diag(&mut self) -> Resultat<Vec<u32>> {
        let diag = self.app.alloc::<u32>(self.nch() * 4 * self.v)?;
        self.hb(diag.arg())?;
        self.app.synchroniser()?;
        let mut h = vec![0u32; diag.len()];
        self.app.lire(&diag, 0, &mut h)?;
        Ok(h)
    }

    fn hb(&mut self, diag: Arg) -> Resultat<()> {
        let cfg = self.lancement_chaines(self.v / 2);
        let u = self.tampon_champ()?.arg();
        let coef = self.coef.as_ref().map_or(Arg::Nul, |c| c.arg());
        let masque = self.arg_masque(self.app_masque.bain());
        for mu in 0..4i32 {
            for p in 0..2i32 {
                self.app.lancer(
                    &self.f_hb,
                    &cfg,
                    &[
                        u,
                        self.geo.arg(),
                        Arg::I32(mu),
                        Arg::I32(p),
                        self.betas.arg(),
                        self.k0s.arg(),
                        self.k1s.arg(),
                        Arg::U32(self.sweep_id),
                        self.cap.arg(),
                        self.tries.arg(),
                        diag,
                        coef,
                        masque,
                    ],
                )?;
            }
        }
        self.sweep_id = self.sweep_id.wrapping_add(1);
        Ok(())
    }
}

impl NoyauxSu2 for Geo64Gpu {
    fn precision(&self) -> Precision {
        Precision::F64
    }
    fn backend(&self) -> BackendKind {
        self.app.backend()
    }
    fn config(&self) -> &ConfigLot {
        &self.cfg
    }
    fn init(&mut self) -> Resultat<()> {
        self.app.lancer(
            &self.f_init,
            &self.lancement_chaines(4 * self.v),
            &[
                self.tampon_champ()?.arg(),
                self.geo.arg(),
                self.hot.arg(),
                self.k0s.arg(),
                self.k1s.arg(),
            ],
        )?;
        self.app.synchroniser()?;
        let zeros = vec![0u64; self.nch()];
        self.app.ecrire(&mut self.cap, 0, &zeros)?;
        self.app.ecrire(&mut self.tries, 0, &zeros)?;
        self.sweep_id = 0;
        Ok(())
    }
    fn fixer_gel(&mut self, gel: Gel) -> Resultat<()> {
        if gel != Gel::Aucun && !self.spec.torsion.est_triviale() {
            return Err(Erreur::Config(
                "gel incompatible avec une torsion de 't Hooft non triviale".into(),
            ));
        }
        self.gel = gel;
        self.recalculer_masques()
    }
    fn fixer_masque(&mut self, masque: Option<&[u8]>, application: AppMasque) -> Resultat<()> {
        self.app_masque = application;
        match masque {
            None => self.masque_ext = None,
            Some(m) => {
                let attendu = 4 * self.v;
                if m.len() != attendu {
                    return Err(Erreur::Config(format!(
                        "masque de {} octets : {attendu} attendus (4·L⁴, L = {})",
                        m.len(),
                        self.cfg.l
                    )));
                }
                if !self.spec.torsion.est_triviale() {
                    return Err(Erreur::Config(
                        "masque de liens gelés incompatible avec une torsion de 't Hooft non triviale".into(),
                    ));
                }
                self.masque_ext = Some(m.to_vec());
            }
        }
        self.recalculer_masques()
    }
    fn invariants_reprise(&self) -> Option<serde_json::Value> {
        Some(invariants_c1(
            self.app.backend().nom(),
            &self.spec,
            self.gel,
            self.masque_ext.as_deref(),
            self.app_masque,
        ))
    }
    fn fixer_torsion(&mut self, t: &Torsion) -> Resultat<()> {
        if *t == self.spec.torsion {
            Ok(())
        } else {
            Err(Erreur::Config(format!(
                "torsion {} différente de celle de la géométrie du générateur ({})",
                t.texte(),
                self.spec.torsion.texte()
            )))
        }
    }
    fn sweep_hb(&mut self) -> Resultat<()> {
        self.hb(Arg::Nul)
    }
    fn sweep_or(&mut self) -> Resultat<()> {
        let cfg = self.lancement_chaines(self.v / 2);
        let u = self.tampon_champ()?.arg();
        let coef = self.coef.as_ref().map_or(Arg::Nul, |c| c.arg());
        let masque = self.arg_masque(self.app_masque.surrelax());
        for mu in 0..4i32 {
            for p in 0..2i32 {
                self.app.lancer(
                    &self.f_or,
                    &cfg,
                    &[u, self.geo.arg(), Arg::I32(mu), Arg::I32(p), coef, masque],
                )?;
            }
        }
        Ok(())
    }
    fn allouer_historique(&mut self, capacite: usize) -> Resultat<()> {
        let t = self.app.alloc::<f64>(self.nch() * capacite * 5)?;
        self.hist = Some((t, capacite));
        self.app.synchroniser()
    }
    fn mesurer(&mut self, idx: usize) -> Resultat<()> {
        let (hist, hcap) = self
            .hist
            .as_ref()
            .ok_or_else(|| Erreur::Config("historique non alloué".into()))?;
        if idx >= *hcap {
            return Err(Erreur::Config(format!(
                "historique plein : idx {idx} ≥ capacité {hcap}"
            )));
        }
        let (v, nch) = (self.v, self.nch());
        let nb = v.div_ceil(BS as usize);
        let out_pas: usize = self.cotes.iter().map(|&l| v / l).sum();
        let u = self.tampon_champ()?.arg();
        let un_bloc = Lancement {
            grille: [1, nch as u32, 1],
            bloc: [BS, 1, 1],
            shmem: 0,
        };
        let pas_hist = Arg::I64((5 * hcap) as i64);
        let app = &self.app;
        let coef = self.coef.as_ref().map_or(Arg::Nul, |c| c.arg());
        app.lancer(
            &self.f_plaq_site,
            &Lancement::lineaire_lot(v, nch, BS),
            &[u, self.geo.arg(), coef, self.site.arg()],
        )?;
        app.lancer(
            &self.f_reduce,
            &Lancement::lineaire_lot(v, nch, BS),
            &[self.site.arg(), Arg::I64(v as i64), self.part.arg()],
        )?;
        app.lancer(
            &self.f_final,
            &un_bloc,
            &[
                self.part.arg(),
                Arg::I64(nb as i64),
                Arg::I64(nb as i64),
                Arg::F64(1.0 / (6.0 * (v as i64) as f64)),
                hist.arg_decale(5 * idx),
                pas_hist,
            ],
        )?;
        let mut off_mu = 0;
        for (mu, &l_mu) in self.cotes.iter().enumerate() {
            let n = v / l_mu;
            app.lancer(
                &self.f_pol,
                &Lancement::lineaire_lot(n, nch, BS),
                &[
                    u,
                    self.geo.arg(),
                    Arg::I32(mu as i32),
                    self.pol.arg_decale(off_mu),
                    Arg::I64(out_pas as i64),
                ],
            )?;
            app.lancer(
                &self.f_final,
                &un_bloc,
                &[
                    self.pol.arg_decale(off_mu),
                    Arg::I64(n as i64),
                    Arg::I64(out_pas as i64),
                    Arg::F64(1.0 / (n as i64) as f64),
                    hist.arg_decale(5 * idx + 1 + mu),
                    pas_hist,
                ],
            )?;
            off_mu += n;
        }
        Ok(())
    }
    fn historique(&self) -> Resultat<Vec<f64>> {
        let (hist, hcap) = self
            .hist
            .as_ref()
            .ok_or_else(|| Erreur::Config("historique non alloué".into()))?;
        self.app.synchroniser()?;
        let mut h = vec![0f64; self.nch() * hcap * 5];
        self.app.lire(hist, 0, &mut h)?;
        Ok(h)
    }
    fn mesurer_tranches(&mut self, _idx: usize, _capacite: usize) -> Resultat<()> {
        Err(Erreur::Config(
            "tranches R2-3 : non définies sur le chemin C1".into(),
        ))
    }
    fn historique_tranches(&self) -> Resultat<Vec<f64>> {
        Err(Erreur::Config(
            "tranches R2-3 : non définies sur le chemin C1".into(),
        ))
    }
    fn telecharger(&self, chaine: usize) -> Resultat<ChampHote> {
        self.app.synchroniser()?;
        self.champ.telecharger(&self.app, chaine)
    }
    fn televerser(&mut self, chaine: usize, c: &ChampHote) -> Resultat<()> {
        self.app.synchroniser()?;
        let app = self.app.clone();
        self.champ.televerser(&app, chaine, c)
    }
    fn champ_gpu(&self) -> Option<&ChampGpu> {
        Some(&self.champ)
    }
    fn compteurs(&self) -> Resultat<Vec<Compteurs>> {
        self.app.synchroniser()?;
        let n = self.nch();
        let (mut cap, mut tries) = (vec![0u64; n], vec![0u64; n]);
        self.app.lire(&self.cap, 0, &mut cap)?;
        self.app.lire(&self.tries, 0, &mut tries)?;
        Ok(cap
            .into_iter()
            .zip(tries)
            .map(|(plafond, essais)| Compteurs { plafond, essais })
            .collect())
    }
    fn sweep_id(&self) -> u32 {
        self.sweep_id
    }
    fn fixer_sweep_id(&mut self, s: u32) {
        self.sweep_id = s;
    }
    fn synchroniser(&self) -> Resultat<()> {
        self.app.synchroniser()
    }
    fn provenance(&self) -> Vec<ProvenanceCompil> {
        vec![self.prov.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn o3_admission_c1_couvre_les_allocations() {
        for l in [4usize, 8, 64, 180] {
            for n in [1usize, 3] {
                let (v, nu) = ((l as u64).pow(4), n as u64);
                // allocations recomptées champ par champ (nouveau_mutant, recalculer_masques) : champ 16V f64, site V f64,
                // réduction ceil(V/256) f64, Polyakov 4·V/L f64, paramètres (β, clés, départ, compteurs : 36 o) par chaîne ;
                // geo 9 i64, coefficients 6V f64, masques 2 × 4V octets.
                let par_chaine = 128 * v + 8 * v + 8 * v.div_ceil(256) + 32 * (v / l as u64) + 36;
                let alloc = nu * par_chaine + 72 + 48 * v + 8 * v;
                let b = besoin_memoire_c1(l, n);
                assert!(
                    b.gpu >= alloc,
                    "L = {l}, {n} chaîne(s) : estimation {} < allocations {alloc}",
                    b.gpu
                );
                assert!(
                    b.hote >= 56 * v,
                    "L = {l} : pic hôte des tables sous-estimé"
                );
                // l'ancienne admission (08d4c84) ne comptait que le C0 : 160V par chaîne, sous les allocations C1
                if l == 64 && n == 1 {
                    assert!(160 * v < alloc, "le défaut d'O3 doit être visible à L = 64");
                }
            }
        }
    }

    #[test]
    fn c09c_p3_admission_du_run_entier_avec_historique() {
        use crate::commun::{Depart, Graine};
        use crate::su2::{ConfigLot, ParamChaine};
        // contre-exemple de la revue c09b : L = 4, une chaîne tordue, therm = 2, nmeas = 1000
        let cfg = crate::su2::run::ConfigRun {
            lot: ConfigLot {
                l: 4,
                chaines: vec![ParamChaine {
                    beta: 2.5,
                    graine: Graine(1),
                    depart: Depart::Chaud,
                }],
                nor: 4,
                precision: Precision::F64,
                options: OptionsCompil::reference_k1(),
                mutant_metro: false,
            },
            therm: 2,
            nmeas: 1000,
            skip: 1,
            dump_every: 0,
            dump_initial: false,
            format_dump: "q64".into(),
            zstd: false,
            reprise_every: 0,
            outdir: "x".into(),
            bord: None,
        };
        let v = 256u64;
        // allocations explicites du run : moteur (voir o3) + historique allouer_historique(max(therm, nmeas) + 1) × 5 f64
        let moteur =
            (128 * v + 8 * v + 8 * v.div_ceil(256) + 32 * (v / 4) + 36) + 72 + 48 * v + 8 * v;
        let historique = 40 * 1001;
        let b = besoin_run_c1(&cfg);
        assert!(
            b.gpu >= moteur + historique,
            "admission GPU {} < allocations {}",
            b.gpu,
            moteur + historique
        );
        assert!(
            b.hote >= historique,
            "rapatriement de l'historique absent de l'hôte"
        );
        // l'admission de 30a35bb (sans historique) restait sous les allocations : 51 792 octets
        let ancienne = crate::su2::run::besoin_memoire(&cfg, true)
            .gpu
            .max(besoin_memoire_c1(4, 1).gpu);
        assert_eq!(ancienne, 51_792);
        assert!(ancienne < moteur + historique);
    }

    #[test]
    fn o4_valeurs_de_bord_prescrites_refusees() {
        let code = |r: Resultat<()>| r.map_err(|e| e.code_sortie());
        assert_eq!(
            code(valider_c1(4, &GeoSpec::hyperplan(4, 3, SourceBord::Froid))),
            Err(2)
        );
        assert_eq!(
            code(valider_c1(
                4,
                &GeoSpec::hyperplan(4, 3, SourceBord::Chaud { graine: 7 })
            )),
            Err(2)
        );
        assert_eq!(
            code(valider_c1(6, &GeoSpec::boite(4, 3, SourceBord::Froid))),
            Err(2)
        );
        assert_eq!(
            code(valider_c1(
                4,
                &GeoSpec::hyperplan(4, 3, SourceBord::ParChaine)
            )),
            Ok(())
        );
        assert_eq!(
            code(valider_c1(6, &GeoSpec::boite(4, 3, SourceBord::ParChaine))),
            Ok(())
        );
    }

    #[test]
    fn tableau_copie_les_tables() {
        let t = Tables::compiler(&GeoSpec::cubique(4, 0)).unwrap();
        assert_eq!(tableau_geo(&t), [4, 4, 4, 4, 256, 64, 16, 4, 1]);
        let t = Tables::compiler(&GeoSpec::anisotrope([2, 4, 6, 8], 0)).unwrap();
        assert_eq!(tableau_geo(&t), [2, 4, 6, 8, 384, 192, 48, 8, 1]);
    }

    #[test]
    fn sources_dans_l_ordre_pour_les_deux_backends() {
        for b in [BackendKind::Hip, BackendKind::Cuda] {
            let src = source_geo64(b, 0);
            let noms: Vec<&str> = src.parties.iter().map(|(n, _)| n.as_str()).collect();
            let prelude = if b == BackendKind::Hip {
                "prelude_hip.h"
            } else {
                "prelude_cuda.h"
            };
            assert_eq!(src.nom, "su2_geo64");
            assert_eq!(
                noms,
                [prelude, "su2_commun.h", "su2_ref64.cu", "su2_geo.cu"]
            );
        }
    }

    #[test]
    fn mutants_aux_empreintes_distinctes() {
        for b in [BackendKind::Hip, BackendKind::Cuda] {
            let empreintes: BTreeSet<String> = (0..=5).map(|m| source_geo64(b, m).sha256).collect();
            assert_eq!(empreintes.len(), 6);
            assert!(source_geo64(b, 2)
                .texte
                .contains("#define KVU_GEO_MUTANT 2"));
        }
    }

    #[test]
    fn memoire_croit_avec_le_lot() {
        for spec in [GeoSpec::cubique(4, 0), GeoSpec::anisotrope([2, 4, 6, 8], 0)] {
            let t = Tables::compiler(&spec).unwrap();
            assert!(memoire_necessaire_geo(&t, 1) < memoire_necessaire_geo(&t, 2));
            assert!(memoire_necessaire_geo(&t, 2) < memoire_necessaire_geo(&t, 4));
        }
    }
}
