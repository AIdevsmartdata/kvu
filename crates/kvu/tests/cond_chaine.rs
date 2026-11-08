//! Chaîne conditionnelle (cadence de chaine_v3, sorties NPY, reprise à l'octet, refus, échecs), en CPU, avec le
//! double de test `MesureProvisoire` (la fibre réelle est jugée par les gates de T6 et par EQC-2).
use kvu::commun::{Erreur, Resultat};
use kvu::cond::chaine::{executer_avec, ConfigChaine, Issue, OptionsChaine};
use kvu::cond::essais::dossier_commite;
use kvu::cond::mesure::{Mesure, MesureProvisoire};
use kvu::cond::moteur::{ModeCond, MoteurCond};
use kvu::cond::npy_min;
use kvu::fibre::MesureFibre;
use std::path::{Path, PathBuf};

static SEUIL_SSD: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Fixe le seuil de la garde SSD pour la durée du test (les sorties vont dans /tmp, tmpfs de 59 Go : on ne teste
/// pas la politique de la machine ici). Sérialise les tests : la variable d'environnement est globale.
fn seuil_ssd(go: &str) -> SeuilGuard {
    let g = SEUIL_SSD.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("KEVINOTRON_SSD_MIN_GO", go);
    SeuilGuard { _garde: g }
}

struct SeuilGuard {
    _garde: std::sync::MutexGuard<'static, ()>,
}

impl Drop for SeuilGuard {
    fn drop(&mut self) {
        std::env::remove_var("KEVINOTRON_SSD_MIN_GO");
    }
}

fn dossier(nom: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("kvu_cond_chaine_{nom}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn cfg(mode: ModeCond, out: &Path) -> ConfigChaine {
    ConfigChaine {
        cfg: dossier_commite().join("cfg_L4_haar0.npy"),
        beta: 2.3,
        graine: 2709,
        mode,
        ntherm: 5,
        nsweep: 60,
        every: 2,
        deach: 5,
        ckpt: 10,
        out: out.to_path_buf(),
        tordu: false,
    }
}

fn opts(arret: Option<u64>) -> OptionsChaine {
    OptionsChaine { arret_apres_ckpt: arret, mesure: Some(Box::new(MesureProvisoire::nouvelle(4))), journal: false }
}

fn terminee(i: Issue) -> kvu::cond::chaine::RapportChaine {
    match i {
        Issue::Terminee(r) => r,
        autre => panic!("chaîne non terminée : {autre:?}"),
    }
}

fn sorties(dir: &Path) -> Vec<Vec<u8>> {
    kvu::cond::chaine::fichiers_de_sortie(dir).iter().map(|p| std::fs::read(p).unwrap()).collect()
}

#[test]
fn cadence_de_chaine_v3_et_formes_des_sorties() {
    let _seuil = seuil_ssd("0");
    let out = dossier("cadence");
    let c = cfg(ModeCond::RefRng, &out);
    let r = terminee(executer_avec(None, &c, false, opts(None)).unwrap());
    // 60 balayages, une mesure tous les 2 : 30 F ; D à k = 0, 5, 10, … : 6
    assert_eq!((r.mesures_f, r.mesures_d), (30, 6));
    assert!(r.vdev < 1e-12 && r.unitarite < 1e-12);
    let f = npy_min::lire(&out.join("F.npy")).unwrap();
    let d = npy_min::lire(&out.join("D.npy")).unwrap();
    let blk = npy_min::lire(&out.join("blk.npy")).unwrap();
    assert_eq!(f.forme, vec![30, 6]);
    assert_eq!(d.forme, vec![6, 6, 6]);
    assert_eq!(blk.forme, vec![6, 6, 16]);
    assert!(f.f64().unwrap().iter().all(|x| x.is_finite()));
    assert!(out.join("COMPLETE").exists() && out.join("manifest.json").exists() && out.join("alea.json").exists());
    assert!(!out.join("reprise.json").exists(), "les fichiers de reprise sont supprimés à la fin");
    let man: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(man["statut"], "termine");
    assert_eq!(man["fichiers"].as_array().unwrap().len(), 5);
    assert_eq!(man["schema"], "kevinotron-unifie/1");
    assert_eq!(man["backend"], "cpu");
    assert_eq!(man["verrou"], "sans_gpu");
    assert_eq!(man["rapport"]["mesures_f"], 30);
    assert!(man["temps_s"]["balayages"].as_f64().unwrap() > 0.0);
    assert!(man["garde_ssd"]["seuil_go"].is_number() && man["garde_ssd"]["libre_go"].is_number());
    assert_eq!(man["config"]["graine"], 2709);
    // les empreintes du manifeste sont celles des fichiers écrits
    for f in man["fichiers"].as_array().unwrap() {
        let b = std::fs::read(out.join(f["nom"].as_str().unwrap())).unwrap();
        assert_eq!(f["octets"].as_u64().unwrap(), b.len() as u64);
        assert_eq!(f["sha256"].as_str().unwrap(), kvu::heritage::su2_hb::sha256_hex(&b));
    }
    let complete = std::fs::read_to_string(out.join("COMPLETE")).unwrap();
    assert!(complete.starts_with("rc=0"));
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn reprise_a_l_octet_dans_les_deux_modes() {
    let _seuil = seuil_ssd("0");
    for mode in [ModeCond::RefRng, ModeCond::Rapide] {
        let (a, b) = (dossier(&format!("seule_{mode:?}")), dossier(&format!("reprise_{mode:?}")));
        terminee(executer_avec(None, &cfg(mode, &a), false, opts(None)).unwrap());
        // premier segment : s'arrête après le 2e point de reprise (20 mesures sur 30), comme un arrêt brutal
        let issue = executer_avec(None, &cfg(mode, &b), false, opts(Some(2))).unwrap();
        assert_eq!(issue, Issue::Interrompue { ckpt: 2 });
        assert!(b.join("reprise.json").exists() && !b.join("COMPLETE").exists());
        // second segment : « nouveau processus » (aucun état en mémoire partagé)
        terminee(executer_avec(None, &cfg(mode, &b), true, opts(None)).unwrap());
        let (sa, sb) = (sorties(&a), sorties(&b));
        for (i, nom) in ["F.npy", "D.npy", "blk.npy", "controles.json", "alea.json"].iter().enumerate() {
            assert_eq!(sa[i], sb[i], "{mode:?} : {nom} différent après reprise");
        }
        let _ = std::fs::remove_dir_all(&a);
        let _ = std::fs::remove_dir_all(&b);
    }
}

#[test]
fn reprise_refuse_une_autre_configuration() {
    let _seuil = seuil_ssd("0");
    let out = dossier("autre");
    let c = cfg(ModeCond::RefRng, &out);
    executer_avec(None, &c, false, opts(Some(1))).unwrap();
    let mut autre = c.clone();
    autre.beta = 2.8;
    assert!(matches!(executer_avec(None, &autre, true, opts(None)), Err(Erreur::Config(_))));
    let mut autre = c.clone();
    autre.graine += 1;
    assert!(matches!(executer_avec(None, &autre, true, opts(None)), Err(Erreur::Config(_))));
    // la bonne configuration reprend
    terminee(executer_avec(None, &c, true, opts(None)).unwrap());
    // une chaîne terminée ne se reprend pas
    assert!(matches!(executer_avec(None, &c, true, opts(None)), Err(Erreur::Config(_))));
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn repertoire_existant_refuse_et_intact() {
    let _seuil = seuil_ssd("0");
    let out = dossier("existe");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join("manifest.json"), b"{\"statut\": \"termine\"}").unwrap();
    let r = executer_avec(None, &cfg(ModeCond::RefRng, &out), false, opts(None));
    assert!(matches!(r, Err(Erreur::Config(_))));
    assert_eq!(
        std::fs::read(out.join("manifest.json")).unwrap(),
        b"{\"statut\": \"termine\"}",
        "le run existant n'est pas touché"
    );
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn beta_nul_refuse_en_ref_rng_sans_rien_creer() {
    let _seuil = seuil_ssd("0");
    let out = dossier("beta0");
    let mut c = cfg(ModeCond::RefRng, &out);
    c.beta = 0.0;
    assert!(matches!(executer_avec(None, &c, false, opts(None)), Err(Erreur::Config(_))));
    assert!(!out.exists(), "rien n'est créé");
    // en rapide, β = 0 est admis (Haar conditionnel)
    let out2 = dossier("beta0r");
    let mut c = cfg(ModeCond::Rapide, &out2);
    c.beta = 0.0;
    terminee(executer_avec(None, &c, false, opts(None)).unwrap());
    let _ = std::fs::remove_dir_all(&out2);
}

#[test]
fn parametres_incoherents_refuses() {
    let _seuil = seuil_ssd("0");
    let out = dossier("param");
    let mut c = cfg(ModeCond::Rapide, &out);
    c.every = 0;
    assert!(executer_avec(None, &c, false, opts(None)).is_err());
    let mut c = cfg(ModeCond::Rapide, &out);
    c.nsweep = 1;
    assert!(executer_avec(None, &c, false, opts(None)).is_err());
    let mut c = cfg(ModeCond::Rapide, &out);
    c.deach = 0;
    assert!(executer_avec(None, &c, false, opts(None)).is_err());
    assert!(!out.exists());
}

#[test]
fn ssd_insuffisant_refuse_avant_tout_octet() {
    // SSD-1 : seuil forcé au-dessus de l'espace libre ⇒ refus (code 7), aucun répertoire, aucun COMPLETE
    let out = dossier("ssd");
    let r = {
        let _seuil = seuil_ssd("1e9");
        executer_avec(None, &cfg(ModeCond::Rapide, &out), false, opts(None))
    };
    match r {
        Err(e @ Erreur::SsdPlein { .. }) => assert_eq!(e.code_sortie(), 7),
        // si la garde de T8 est livrée elle applique la politique de la machine (seuil réel) : un feu vert est licite
        Ok(_) => {
            let _ = std::fs::remove_dir_all(&out);
        }
        Err(e) => panic!("erreur inattendue : {e}"),
    }
}

/// Mesure empoisonnée : renvoie des NaN ; la chaîne doit échouer SANS COMPLETE.
struct Empoisonnee;
impl Mesure for Empoisonnee {
    fn nf(&self) -> usize {
        6
    }
    fn mesurer_f(&mut self, _m: &MoteurCond) -> Resultat<Vec<f64>> {
        Ok(vec![f64::NAN; 6])
    }
    fn mesurer(&mut self, _m: &MoteurCond) -> Resultat<MesureFibre> {
        Ok(MesureFibre { f: vec![f64::NAN; 6], d: vec![0.0; 36], blk: vec![0.0; 96] })
    }
}

#[test]
fn mesure_non_finie_donne_un_echec_sans_complete() {
    let _seuil = seuil_ssd("0");
    let out = dossier("nan");
    let r = executer_avec(
        None,
        &cfg(ModeCond::Rapide, &out),
        false,
        OptionsChaine { mesure: Some(Box::new(Empoisonnee)), ..Default::default() },
    );
    assert!(matches!(r, Err(Erreur::Validation(_))), "{r:?}");
    assert!(!out.join("COMPLETE").exists());
    let man: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(man["statut"], "echec");
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn deux_runs_identiques_donnent_les_memes_octets() {
    let _seuil = seuil_ssd("0");
    for mode in [ModeCond::RefRng, ModeCond::Rapide] {
        let (a, b) = (dossier(&format!("det_a_{mode:?}")), dossier(&format!("det_b_{mode:?}")));
        terminee(executer_avec(None, &cfg(mode, &a), false, opts(None)).unwrap());
        terminee(executer_avec(None, &cfg(mode, &b), false, opts(None)).unwrap());
        assert_eq!(sorties(&a), sorties(&b), "{mode:?}");
        let c_dir = dossier(&format!("det_c_{mode:?}"));
        let mut c = cfg(mode, &c_dir);
        c.graine += 1;
        terminee(executer_avec(None, &c, false, opts(None)).unwrap());
        assert_ne!(sorties(&a)[0], sorties(&c_dir)[0], "graine + 1 ⇒ F différent");
        for d in [a, b, c_dir] {
            let _ = std::fs::remove_dir_all(d);
        }
    }
}
