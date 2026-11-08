//! Sous-commande `groupe` : registre des groupes de jauge et batterie à réponse connue (WP5). Propriétaire : WP5.
//!
//! ```text
//! kvu groupe liste
//! kvu groupe gates --groupe su2 [--backend cpu|auto|hip|cuda] [--json] [--seul ID[,ID…]] [--sans-mutants]
//!                  [--tirages N] [--certifier [DIR]]
//! kvu groupe certificat --groupe X [--dossier DIR]
//! kvu groupe derive --groupe su2 [--tirages N]
//! kvu groupe plan --groupe sp4 --dossier DIR [--binaire CHEMIN] [--sans-mutants] [--sans-info]
//! kvu groupe chaine --groupe sp4 --id ID --dossier DIR [--mesures N] [--reprise N]
//! ```
//! Sp(4) (WP16) : `gates --groupe sp4 [--chaines DIR] [--sans-calcul]` ; les chaînes Monte Carlo (G-SC, G-WC, G-REG, mutants) sont des séries lues dans DIR,
//! produites en parallèle par `chaine` (lignes données par `plan`), ou calculées en série par `gates` si `--sans-calcul` est absent.
//! `gates` émet une ligne `PASS|FAIL|SKIP|INFO <ID> <détail>` par gate (stdout), et avec `--json` un document JSON final.
//! Code de sortie : 0 si aucune ligne FAIL ni SKIP, 8 sinon (donc 8 sur CPU tant que `G-UNIT-F32` est SKIP, même si le certificat `cpu` est écrit),
//! 2 si le groupe n'est pas au registre.
//! `--certifier [DIR]` (défaut `validation/groupes`) écrit `DIR/<groupe>.json` (horodatage, sha256 du binaire, verdicts) et refuse
//! de l'écrire si une ligne est FAIL ou SKIP (seul le SKIP prescrit `G-UNIT-F32` est admis dans l'entrée `cpu`, règle 0.2 ; G-CERT). `--groupe X` est REFUSÉ si X n'a ni fiche ni certificat.

use super::batterie_generique::{self, Calcul, ContexteGen, ProtoGen};
use super::gates::{self, certifiable, tout_vert, Contexte, LigneGate, Protocole, Verdict};
use super::{trouver, Groupe};
use crate::commun::{Erreur, Resultat};
use crate::gpu::Appareil;
use clap::{Arg, ArgAction, ArgMatches, Command};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const SCHEMA_CERT: &str = "kvu-groupe-cert/1";
pub const DOSSIER_CERT_DEFAUT: &str = "validation/groupes";

pub fn commandes() -> Vec<Command> {
    vec![super::gen_cli::commande(), Command::new("groupe")
        .about("Registre des groupes de jauge : fiche, batterie de gates à réponse connue, certificat")
        .subcommand_required(true)
        .subcommand(Command::new("liste").about("Groupes au registre (avec fiche)"))
        .subcommand(
            Command::new("gates")
                .about("Batterie à réponse connue d'un groupe : une ligne PASS|FAIL|SKIP par gate")
                .arg(Arg::new("groupe").long("groupe").required(true).num_args(1).help("su2"))
                .arg(Arg::new("backend").long("backend").num_args(1).default_value("cpu").help("cpu | auto | hip | cuda (GPU : sous le verrou GPU)"))
                .arg(Arg::new("json").long("json").action(ArgAction::SetTrue).help("ajoute un document JSON final"))
                .arg(Arg::new("seul").long("seul").num_args(1).help("n'exécute que les gates dont l'ID commence par l'un de ces préfixes (liste à virgules)"))
                .arg(Arg::new("sans-mutants").long("sans-mutants").action(ArgAction::SetTrue).help("omet G-MUT (développement ; le certificat l'exige)"))
                .arg(Arg::new("tirages").long("tirages").num_args(1).help("tirages de Haar (défaut gelé : 1000000)"))
                .arg(Arg::new("certifier").long("certifier").num_args(0..=1).default_missing_value(DOSSIER_CERT_DEFAUT).help("écrit le certificat DIR/<groupe>.json (défaut validation/groupes)"))
                .arg(Arg::new("chaines").long("chaines").num_args(1).help("sp4 : dossier des séries de chaînes (défaut : ./chaines_<groupe>)"))
                .arg(Arg::new("sans-calcul").long("sans-calcul").action(ArgAction::SetTrue).help("sp4 : ne simule rien ; une série absente ou trop courte donne SKIP")),
        )
        .subcommand(
            Command::new("plan")
                .about("Sp(4) : une commande `chaine` par ligne pour l’exécution en lot, plus l'estimation de coût sur stderr")
                .arg(Arg::new("groupe").long("groupe").required(true).num_args(1))
                .arg(Arg::new("dossier").long("dossier").required(true).num_args(1))
                .arg(Arg::new("binaire").long("binaire").num_args(1).default_value("kvu"))
                .arg(Arg::new("sans-mutants").long("sans-mutants").action(ArgAction::SetTrue))
                .arg(Arg::new("sans-info").long("sans-info").action(ArgAction::SetTrue).help("omet les chaînes des lignes INFO (β × 1,002)")),
        )
        .subcommand(
            Command::new("chaine")
                .about("Sp(4) : exécute UNE chaîne de la batterie (ou UNIT) avec points de reprise ; écrit DIR/<id>.serie")
                .arg(Arg::new("groupe").long("groupe").required(true).num_args(1))
                .arg(Arg::new("id").long("id").required(true).num_args(1))
                .arg(Arg::new("dossier").long("dossier").required(true).num_args(1))
                .arg(Arg::new("mesures").long("mesures").num_args(1).help("prolonge/force le nombre de pas mesurés (extension v4.0 : mêmes graine et ε)"))
                .arg(Arg::new("reprise").long("reprise").num_args(1).default_value("1000").help("point de reprise tous les N pas (0 = seulement à la fin)")),
        )
        .subcommand(
            Command::new("certificat")
                .about("Vérifie qu'un groupe a un certificat vert (G-CERT) ; refus (code 2) sinon")
                .arg(Arg::new("groupe").long("groupe").required(true).num_args(1))
                .arg(Arg::new("dossier").long("dossier").num_args(1).default_value(DOSSIER_CERT_DEFAUT)),
        )
        .subcommand(
            Command::new("derive")
                .about("Intégration de Haar numérique des dérivations de v3.1 (pente, u(β)) ; information, pas un verdict")
                .arg(Arg::new("groupe").long("groupe").required(true).num_args(1))
                .arg(Arg::new("tirages").long("tirages").num_args(1).default_value("1000000")),
        )]
}

pub fn executer(nom: &str, m: &ArgMatches) -> Resultat<i32> {
    if nom == "gen" {
        return super::gen_cli::executer(m);
    }
    if nom != "groupe" {
        return Err(Erreur::Config(format!("sous-commande « {nom} » inconnue du module groupe")));
    }
    match m.subcommand() {
        Some(("liste", _)) => {
            for n in super::noms_registre() {
                println!("{n}");
            }
            Ok(0)
        }
        Some(("gates", s)) => gates_cmd(s),
        Some(("certificat", s)) => {
            let g = texte(s, "groupe")?;
            let dossier = PathBuf::from(texte(s, "dossier")?);
            exiger_certificat(&g, &dossier)?;
            println!("PASS G-CERT certificat valide : {}", chemin_certificat(&dossier, &g).display());
            Ok(0)
        }
        Some(("derive", s)) => derive_cmd(s),
        Some(("plan", s)) => plan_cmd(s),
        Some(("chaine", s)) => chaine_cmd(s),
        _ => Err(Erreur::Config("sous-commande de `groupe` manquante".into())),
    }
}

fn texte(m: &ArgMatches, nom: &str) -> Resultat<String> {
    m.get_one::<String>(nom).cloned().ok_or_else(|| Erreur::Config(format!("--{nom} requis")))
}

fn groupe_ou_refus(nom: &str) -> Resultat<Box<dyn Groupe>> {
    trouver(nom).ok_or_else(|| {
        Erreur::Config(format!(
            "groupe « {nom} » refusé : aucune fiche au registre ({}) et aucun certificat de gates ({DOSSIER_CERT_DEFAUT}/{nom}.json) ; G-CERT",
            super::noms_registre().join(", ")
        ))
    })
}

// ------------------------------------------------------------------------------------------
// Certificat (G-CERT)
// ------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntreeCertificat {
    pub backend: String,
    pub horodatage: String,
    pub binaire_sha256: String,
    pub version: String,
    /// Aucune ligne FAIL, aucun SKIP hors `skips_prescrits`.
    pub tout_vert: bool,
    /// SKIP prescrits par la règle 0.2 (CPU : G-UNIT-F32, à rejouer en PASS sur GPU) ; vide pour un backend GPU.
    #[serde(default)]
    pub skips_prescrits: Vec<String>,
    pub verdicts: Vec<LigneJson>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LigneJson {
    pub verdict: String,
    pub id: String,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Certificat {
    pub schema: String,
    pub groupe: String,
    pub certificats: Vec<EntreeCertificat>,
}

pub fn chemin_certificat(dossier: &Path, groupe: &str) -> PathBuf {
    dossier.join(format!("{groupe}.json"))
}

/// Refuse (Erreur::Config, code 2) si le groupe n'a pas de certificat valide : fichier lisible, schéma, entrée `cpu` verte.
pub fn exiger_certificat(groupe: &str, dossier: &Path) -> Resultat<Certificat> {
    let p = chemin_certificat(dossier, groupe);
    let txt = std::fs::read_to_string(&p).map_err(|_| {
        Erreur::Config(format!("groupe « {groupe} » refusé : pas de certificat ({} absent) ; G-CERT", p.display()))
    })?;
    let c: Certificat = serde_json::from_str(&txt).map_err(|e| Erreur::Format(format!("{} : {e}", p.display())))?;
    if c.schema != SCHEMA_CERT || c.groupe != groupe {
        return Err(Erreur::Config(format!("certificat {} : schéma ou groupe inattendu ({}, {})", p.display(), c.schema, c.groupe)));
    }
    match c.certificats.iter().find(|e| e.backend == "cpu") {
        Some(e) if e.tout_vert => Ok(c),
        Some(_) => Err(Erreur::Config(format!("groupe « {groupe} » refusé : l'entrée cpu de {} n'est pas verte", p.display()))),
        None => Err(Erreur::Config(format!("groupe « {groupe} » refusé : pas d'entrée cpu dans {}", p.display()))),
    }
}

fn sha_binaire() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| crate::heritage::manifest::sha256_fichier(&p).ok())
        .unwrap_or_else(|| "inconnu".into())
}

fn entree(backend: &str, lignes: &[LigneGate]) -> EntreeCertificat {
    let admis = certifiable(lignes, backend);
    EntreeCertificat {
        backend: backend.to_string(),
        horodatage: crate::regen::manifeste::maintenant_iso(),
        binaire_sha256: sha_binaire(),
        version: crate::cli::version_texte().replace('\n', " | "),
        tout_vert: admis.is_some(),
        skips_prescrits: admis.unwrap_or_default(),
        verdicts: lignes.iter().map(|l| LigneJson { verdict: l.verdict.to_string(), id: l.id.clone(), detail: l.detail.clone() }).collect(),
    }
}

/// Écrit (ou met à jour l'entrée du backend dans) `dossier/<groupe>.json`. Refuse si une ligne n'est ni PASS ni INFO.
pub fn ecrire_certificat(groupe: &str, dossier: &Path, backend: &str, lignes: &[LigneGate]) -> Resultat<PathBuf> {
    if certifiable(lignes, backend).is_none() {
        return Err(Erreur::Validation("certificat refusé : au moins une ligne FAIL ou SKIP non prescrit pour ce backend".into()));
    }
    let p = chemin_certificat(dossier, groupe);
    let mut c = std::fs::read_to_string(&p)
        .ok()
        .and_then(|t| serde_json::from_str::<Certificat>(&t).ok())
        .filter(|c| c.schema == SCHEMA_CERT && c.groupe == groupe)
        .unwrap_or(Certificat { schema: SCHEMA_CERT.into(), groupe: groupe.into(), certificats: vec![] });
    c.certificats.retain(|e| e.backend != backend);
    c.certificats.push(entree(backend, lignes));
    std::fs::create_dir_all(dossier).map_err(|e| Erreur::io("création", e))?;
    let json = serde_json::to_string_pretty(&c).map_err(|e| Erreur::Format(e.to_string()))?;
    std::fs::write(&p, json + "\n").map_err(|e| Erreur::io(format!("écriture {}", p.display()), e))?;
    Ok(p)
}

// ------------------------------------------------------------------------------------------
// gates
// ------------------------------------------------------------------------------------------

fn ouvrir(backend: &str) -> Resultat<Option<Appareil>> {
    crate::su2::cli::ouvrir_backend(backend)
}

fn gates_cmd(m: &ArgMatches) -> Resultat<i32> {
    let nom = texte(m, "groupe")?;
    let _g = groupe_ou_refus(&nom)?;
    let mut protocole = Protocole::gele();
    if let Some(t) = m.get_one::<String>("tirages") {
        protocole.n_haar = t.parse().map_err(|_| Erreur::Config(format!("--tirages « {t} » invalide")))?;
    }
    if m.get_flag("sans-mutants") {
        protocole.mutants = false;
    }
    let seul: Vec<String> = m.get_one::<String>("seul").map(|s| s.split(',').map(|x| x.trim().to_string()).collect()).unwrap_or_default();
    if m.get_one::<String>("certifier").is_some() && (!seul.is_empty() || !protocole.mutants || protocole.n_haar != Protocole::gele().n_haar) {
        return Err(Erreur::Config("--certifier exige la batterie complète et gelée (ni --seul, ni --sans-mutants, ni --tirages)".into()));
    }
    if nom == "sp4" {
        return gates_sp4_cmd(m, &nom, &protocole, seul);
    }
    let backend_demande = m.get_one::<String>("backend").map(|s| s.as_str()).unwrap_or("cpu");
    let app = ouvrir(backend_demande)?;
    // Sur GPU : verrou GPU de la machine, tenu pendant toute la batterie (règle 0.7 de GATES.md).
    let _verrou = if app.is_some() {
        let profil = crate::machine::detection::detecter_rapide();
        let pol = crate::machine::politique(&profil);
        Some(crate::machine::VerrouGpu::nouveau(&pol.verrou_gpu).acquerir(std::time::Duration::from_secs(pol.attente_verrou_s))?)
    } else {
        None
    };
    let ctx = Contexte { app: app.as_ref(), protocole, seul };
    let t0 = std::time::Instant::now();
    let lignes = match nom.as_str() {
        "su2" => gates::batterie_su2(&ctx)?,
        autre => return Err(Erreur::Config(format!("groupe « {autre} » : pas de batterie"))),
    };
    for l in &lignes {
        println!("{l}");
    }
    let (np, nf, ns) = compte(&lignes);
    println!("INFO G-BILAN {nom} backend {} : {np} PASS, {nf} FAIL, {ns} SKIP, {:.1} s", ctx.backend(), t0.elapsed().as_secs_f64());
    if m.get_flag("json") {
        let doc = serde_json::json!({
            "groupe": nom, "backend": ctx.backend(), "tout_vert": tout_vert(&lignes),
            "pass": np, "fail": nf, "skip": ns, "lignes": lignes,
        });
        println!("{}", serde_json::to_string(&doc).map_err(|e| Erreur::Format(e.to_string()))?);
    }
    if let Some(dossier) = m.get_one::<String>("certifier") {
        let p = ecrire_certificat(&nom, Path::new(dossier), ctx.backend(), &lignes)?;
        println!("INFO G-CERT certificat écrit : {}", p.display());
    }
    Ok(if nf == 0 && ns == 0 { 0 } else { crate::commun::code::VALIDATION })
}

fn protocole_sp4(protocole: &Protocole) -> ProtoGen {
    let mut p = ProtoGen::gele();
    p.n_haar = protocole.n_haar;
    p.mutants = protocole.mutants;
    p
}

fn exiger_groupe_sp4(m: &ArgMatches) -> Resultat<()> {
    let nom = texte(m, "groupe")?;
    if nom != "sp4" {
        return Err(Erreur::Config(format!("groupe « {nom} » : pas de chaînes génériques (sp4 seulement)")));
    }
    Ok(())
}

fn gates_sp4_cmd(m: &ArgMatches, nom: &str, protocole: &Protocole, seul: Vec<String>) -> Resultat<i32> {
    let backend = m.get_one::<String>("backend").map(|s| s.as_str()).unwrap_or("cpu");
    if backend != "cpu" {
        return Err(Erreur::Config(format!("sp4 : CPU seulement dans ce lot (--backend {backend} refusé ; GPU = WP17)")));
    }
    let dossier = m.get_one::<String>("chaines").cloned().unwrap_or_else(|| format!("chaines_{nom}"));
    let ctx = ContexteGen {
        protocole: protocole_sp4(protocole),
        seul,
        calcul: Calcul { dossier: PathBuf::from(dossier), calculer: !m.get_flag("sans-calcul"), pas_reprise: 1000 },
    };
    let t0 = std::time::Instant::now();
    let lignes = batterie_generique::batterie_sp4(&ctx)?;
    for l in &lignes {
        println!("{l}");
    }
    let (np, nf, ns) = compte(&lignes);
    println!("INFO G-BILAN {nom} backend cpu : {np} PASS, {nf} FAIL, {ns} SKIP, {:.1} s", t0.elapsed().as_secs_f64());
    if m.get_flag("json") {
        let doc = serde_json::json!({
            "groupe": nom, "backend": "cpu", "tout_vert": tout_vert(&lignes),
            "pass": np, "fail": nf, "skip": ns, "lignes": lignes,
        });
        println!("{}", serde_json::to_string(&doc).map_err(|e| Erreur::Format(e.to_string()))?);
    }
    if let Some(dossier) = m.get_one::<String>("certifier") {
        let p = ecrire_certificat(nom, Path::new(dossier), "cpu", &lignes)?;
        println!("INFO G-CERT certificat écrit : {}", p.display());
    }
    Ok(if nf == 0 && ns == 0 { 0 } else { crate::commun::code::VALIDATION })
}

fn plan_cmd(m: &ArgMatches) -> Resultat<i32> {
    exiger_groupe_sp4(m)?;
    let mut p = ProtoGen::gele();
    p.mutants = !m.get_flag("sans-mutants");
    let dossier = texte(m, "dossier")?;
    let binaire = texte(m, "binaire")?;
    let mut cat = batterie_generique::catalogue(&p);
    if m.get_flag("sans-info") {
        cat.retain(|(_, essentiel)| *essentiel);
    }
    // les plus coûteuses d'abord (L⁴ × pas totaux)
    let cout = |s: &super::generique::SpecChaine| (s.l as f64).powi(4) * (s.therm + s.mesures) as f64;
    cat.sort_by(|a, b| cout(&b.0).partial_cmp(&cout(&a.0)).unwrap());
    let total: f64 = cat.iter().map(|(s, _)| cout(s)).sum::<f64>() + (p.l_unit as f64).powi(4) * p.pas_unit_total as f64;
    println!("{binaire} groupe chaine --groupe sp4 --id UNIT --dossier {dossier}");
    for (s, _) in &cat {
        println!("{binaire} groupe chaine --groupe sp4 --id {} --dossier {dossier}", s.id);
    }
    eprintln!("INFO plan : {} chaînes + UNIT, coût total {total:.3e} (sites × pas) ; à 12 µs par lien et par pas : {:.1} heures-cœur", cat.len(), total * 4.0 * 12e-6 / 3600.0);
    Ok(0)
}

fn chaine_cmd(m: &ArgMatches) -> Resultat<i32> {
    exiger_groupe_sp4(m)?;
    let p = ProtoGen::gele();
    let id = texte(m, "id")?;
    let dossier = PathBuf::from(texte(m, "dossier")?);
    let reprise: u64 = texte(m, "reprise")?.parse().map_err(|_| Erreur::Config("--reprise invalide".into()))?;
    let g = super::sp4::Sp4::nouveau();
    let t0 = std::time::Instant::now();
    if id == "UNIT" {
        let c = Calcul { dossier, calculer: true, pas_reprise: reprise };
        let r = batterie_generique::calculer_unit(&g, &c, &p)?.ok_or_else(|| Erreur::Config("UNIT : calcul impossible".into()))?;
        println!("INFO chaine UNIT : défaut {:.3e}, avant ré-unitarisation {:.3e}, ε = {:.6}, {:.1} s", r.defaut, r.avant_reproj, r.eps, t0.elapsed().as_secs_f64());
        return Ok(0);
    }
    let mesures = match m.get_one::<String>("mesures") {
        Some(t) => Some(t.parse::<u64>().map_err(|_| Erreur::Config("--mesures invalide".into()))?),
        None => None,
    };
    let spec = batterie_generique::spec_par_id(&p, &id, mesures).ok_or_else(|| Erreur::Config(format!("chaîne « {id} » inconnue de la batterie sp4")))?;
    let s = super::generique::executer_chaine(&g, &spec, &dossier, reprise)?;
    println!("INFO chaine {id} : {} valeurs, ε = {:.6}, L = {}, β simulé = {}, graine {}, {:.1} s", s.valeurs.len(), s.eps, spec.l, spec.beta, spec.graine, t0.elapsed().as_secs_f64());
    Ok(0)
}

fn compte(l: &[LigneGate]) -> (usize, usize, usize) {
    let c = |v: Verdict| l.iter().filter(|x| x.verdict == v).count();
    (c(Verdict::Pass), c(Verdict::Fail), c(Verdict::Skip))
}

// ------------------------------------------------------------------------------------------
// derive : canal « Haar numérique » de la dérivation v3.1
// ------------------------------------------------------------------------------------------

fn derive_cmd(m: &ArgMatches) -> Resultat<i32> {
    let nom = texte(m, "groupe")?;
    let g = groupe_ou_refus(&nom)?;
    let n: u64 = texte(m, "tirages")?.parse().map_err(|_| Erreur::Config("--tirages invalide".into()))?;
    let (m2, e2, v) = gates::haar_pente(g.as_ref(), n, gates::GRAINE_DERIVATION_HAAR, &gates::BETAS_SC);
    let d = g.d_f() as f64;
    println!("INFO DERIVE-HAAR {nom} n = {n}, graine {} : ⟨x²⟩ = {m2:.6} ± {e2:.1e} (x = Re Tr U/d_F ; 1/d_F² = {:.6})", gates::GRAINE_DERIVATION_HAAR, 1.0 / (d * d));
    if nom == "su2" {
        for (b, u, s) in &v {
            let ub = gates::u_bessel(*b);
            println!("INFO DERIVE-HAAR u({b}) = ⟨x⟩_β (repondération) = {u:.7} ± {s:.1e} ; I₂/I₁ = {ub:.10} ; écart {:+.2}σ", (u - ub) / s);
        }
        let a_num = gates::a_ajuste(gates::BETAS_SC[0], v[0].1, v[1].1);
        let s_a = (16.0 * v[0].2.powi(2) + v[1].2.powi(2)).sqrt() / (2.0 * gates::BETAS_SC[0]);
        let a_u = gates::a_ajuste(gates::BETAS_SC[0], gates::u_bessel(gates::BETAS_SC[0]), gates::u_bessel(gates::BETAS_SC[1]));
        println!("INFO DERIVE-HAAR a (ajustement exact sur u seul, Haar numérique) = {a_num:.6} ± {s_a:.1e} ; même ajustement sur I₂/I₁ exact = {a_u:.10} ; écart {:+.2}σ", (a_num - a_u) / s_a);
        println!("INFO DERIVE-HAAR a_attendu (u + 4u⁵, v3.1) recalculé = {:.12}", gates::a_attendu_recalcule());
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g2_refuse_sans_certificat() {
        // WP16 : sp4 a une fiche ; g2 reste refusé (assertion existante déplacée de sp4 à g2, signalée au journal)
        let e = groupe_ou_refus("g2").err().expect("g2 doit être refusé");
        assert_eq!(e.code_sortie(), crate::commun::code::USAGE);
        assert!(e.to_string().contains("refusé"));
        let rep = std::env::temp_dir().join("kvu_wp5_cert_absent");
        assert!(exiger_certificat("g2", &rep).is_err());
        assert!(exiger_certificat("sp4", &rep).is_err());
    }

    #[test]
    fn certificat_refuse_si_non_vert_et_roundtrip_si_vert() {
        let rep = std::env::temp_dir().join(format!("kvu_wp5_cert_{}", std::process::id()));
        let rouge = vec![LigneGate::nouvelle(Verdict::Fail, "G-X", "rouge")];
        assert!(ecrire_certificat("su2", &rep, "cpu", &rouge).is_err());
        let skip = vec![LigneGate::nouvelle(Verdict::Skip, "G-X", "skip")];
        assert!(ecrire_certificat("su2", &rep, "cpu", &skip).is_err());
        assert!(exiger_certificat("su2", &rep).is_err());
        // le SKIP prescrit G-UNIT-F32 est admis sur cpu (et noté dans l'entrée), jamais sur un backend GPU
        let skip_prescrit = vec![LigneGate::nouvelle(Verdict::Pass, "G-X", "ok"), LigneGate::nouvelle(Verdict::Skip, "G-UNIT-F32", "gpu")];
        assert!(ecrire_certificat("su2", &rep, "hip", &skip_prescrit).is_err());
        ecrire_certificat("su2", &rep, "cpu", &skip_prescrit).unwrap();
        assert_eq!(exiger_certificat("su2", &rep).unwrap().certificats[0].skips_prescrits, vec!["G-UNIT-F32".to_string()]);
        let vert = vec![LigneGate::nouvelle(Verdict::Pass, "G-X", "ok"), LigneGate::nouvelle(Verdict::Info, "G-Y", "i")];
        let p = ecrire_certificat("su2", &rep, "cpu", &vert).unwrap();
        assert!(p.is_file());
        let c = exiger_certificat("su2", &rep).unwrap();
        assert_eq!(c.certificats.len(), 1);
        // l'entrée d'un autre backend s'ajoute, celle du même backend est remplacée
        ecrire_certificat("su2", &rep, "hip", &vert).unwrap();
        ecrire_certificat("su2", &rep, "cpu", &vert).unwrap();
        assert_eq!(exiger_certificat("su2", &rep).unwrap().certificats.len(), 2);
        let _ = std::fs::remove_dir_all(&rep);
    }
}
