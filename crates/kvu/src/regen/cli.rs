// SPDX-License-Identifier: Apache-2.0
//! Sous-commande `regen` et sous-commande `store classer` (simulation).
//!
//! ```text
//! kvu regen M.json [--comparer DIR] [--sortie DIR] [--ancres anchors.sha256] [--preuve-gbit FICHIER] [--delai S]
//! kvu regen M.json --adapter-seulement --vers FICHIER.json [--hacher] [--preuve-gbit FICHIER]
//! kvu regen --adapter-ensemble ENSEMBLE --vers DIR [--hacher] [--preuve-gbit FICHIER]
//! kvu store classer --chemin ENSEMBLE [--chemin …] [--preuve-gbit FICHIER] [--detail] [--json]
//! ```
//! Codes de sortie : 0 identique / valide, 8 différence ou manifeste invalide, 2 usage (binaire non admis, sortie interdite),
//! 5 délai dépassé, 6 verrou, 7 SSD. `store classer` ne modifie rien ; `regen` n'écrit que dans son répertoire de sortie NEUF.

use super::adaptateurs::{adapter, Hachage};
use super::classer::{Classe, Contexte};
use super::ensemble::{classer_ensemble, trouver_chaines};
use super::manifeste::{est_sous, sha_fichier};
use super::rejeu::{rejouer, OptionsRejeu};
use super::schema::valider_manifeste;
use crate::commun::{Erreur, Resultat};
use crate::stockage::cli::taille_lisible;
use clap::{Arg, ArgAction, ArgMatches, Command};
use std::path::{Path, PathBuf};

pub fn commandes() -> Vec<Command> {
    vec![commande_regen()]
}

fn commande_regen() -> Command {
    Command::new("regen")
        .about("Régénère une chaîne depuis son manifeste (su2_hip/1, su2_gen/1, kvu-regen/1) et compare les sha256 fichier par fichier")
        .arg(Arg::new("manifeste").value_name("M.json").help("manifeste de la chaîne (lu tel quel, jamais modifié)"))
        .arg(Arg::new("comparer").long("comparer").value_name("DIR").help("répertoire des originaux (défaut : celui du manifeste)"))
        .arg(Arg::new("sortie").long("sortie").value_name("DIR").help("répertoire de sortie NEUF (défaut : $KVU_SCRATCH/regen/<nom>_<date>)"))
        .arg(Arg::new("ancres").long("ancres").value_name("FICHIER").help("tests/anchors.sha256 : comparaison supplémentaire aux ancres"))
        .arg(Arg::new("preuve-gbit").long("preuve-gbit").value_name("FICHIER").help("preuves G-BIT (lignes « PASS <ID> binaire_sha256=<sha> … ») pour ce binaire"))
        .arg(Arg::new("delai").long("delai").value_name("S").help("délai maximal du rejeu (défaut : fenêtre GPU de la politique, 600 s)"))
        .arg(Arg::new("mutant-graine").long("mutant-graine").action(ArgAction::SetTrue).hide(true).help("contrôle négatif REGEN-3 : graine + 1"))
        .arg(Arg::new("adapter-seulement").long("adapter-seulement").action(ArgAction::SetTrue).help("traduit en kvu-regen/1 sans rejouer (écrit --vers)"))
        .arg(Arg::new("adapter-ensemble").long("adapter-ensemble").value_name("ENSEMBLE").help("traduit TOUTES les chaînes d'un ensemble en kvu-regen/1 dans --vers (sans rejouer)"))
        .arg(Arg::new("vers").long("vers").value_name("CHEMIN").help("destination des manifestes traduits (NEUVE)"))
        .arg(Arg::new("hacher").long("hacher").action(ArgAction::SetTrue).help("calcule les sha256 des fichiers quand le manifeste d'origine n'en porte pas (su2_hip)"))
}

/// Sous-commande `classer` de `store` (insérée par `stockage::cli`).
pub fn commande_classer() -> Command {
    Command::new("classer")
        .about("SIMULATION : classe A/B/C chaque chaîne des ensembles (aucun fichier modifié, déplacé ni supprimé)")
        .arg(Arg::new("chemin").long("chemin").value_name("ENSEMBLE").action(ArgAction::Append).required(true).help("répertoire d'ensemble, en LECTURE seule"))
        .arg(Arg::new("preuve-gbit").long("preuve-gbit").value_name("FICHIER").help("preuves G-BIT valables pour CE binaire"))
        .arg(Arg::new("detail").long("detail").action(ArgAction::SetTrue).help("une ligne par chaîne"))
        .arg(Arg::new("json").long("json").action(ArgAction::SetTrue))
}

pub fn executer(nom: &str, m: &ArgMatches) -> Resultat<i32> {
    match nom {
        "regen" => executer_regen(m),
        autre => Err(Erreur::Config(format!("sous-commande de régénération inconnue « {autre} »"))),
    }
}

fn contexte_avec_preuves(m: &ArgMatches) -> Resultat<Contexte> {
    let mut ctx = Contexte::nouveau();
    if let Some(p) = m.get_one::<String>("preuve-gbit") {
        let exe = std::env::current_exe().map_err(|e| Erreur::io("chemin du binaire courant", e))?;
        ctx.charger_preuves(Path::new(p), &sha_fichier(&exe)?)?;
    }
    Ok(ctx)
}

fn executer_regen(m: &ArgMatches) -> Resultat<i32> {
    if let Some(ens) = m.get_one::<String>("adapter-ensemble") {
        return adapter_ensemble(Path::new(ens), m);
    }
    let man = m
        .get_one::<String>("manifeste")
        .ok_or_else(|| Erreur::Config("regen : manifeste requis (M.json) ou --adapter-ensemble ENSEMBLE".into()))?;
    let man = PathBuf::from(man);
    if m.get_flag("adapter-seulement") {
        let vers = m.get_one::<String>("vers").ok_or_else(|| Erreur::Config("--adapter-seulement exige --vers FICHIER".into()))?;
        return adapter_un(&man, Path::new(vers), m);
    }
    let sortie = match m.get_one::<String>("sortie") {
        Some(s) => PathBuf::from(s),
        None => {
            let base = std::env::var_os("KVU_SCRATCH")
                .map(PathBuf::from)
                .ok_or_else(|| Erreur::Config("regen : --sortie DIR requis (ou KVU_SCRATCH défini)".into()))?;
            let nom = man.parent().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "chaine".into());
            base.join("regen").join(format!("{nom}_{}", std::process::id()))
        }
    };
    let delai = m
        .get_one::<String>("delai")
        .map(|s| s.parse::<u64>().map_err(|_| Erreur::Config("--delai : entier attendu".into())))
        .transpose()?;
    let opts = OptionsRejeu {
        comparer: m.get_one::<String>("comparer").map(PathBuf::from),
        sortie,
        ancres: m.get_one::<String>("ancres").map(PathBuf::from),
        preuves: m.get_one::<String>("preuve-gbit").map(PathBuf::from),
        delai_s: delai,
        mutant_graine: m.get_flag("mutant-graine"),
        executant: None,
    };
    let r = rejouer(&man, &opts)?;
    println!("regen : {}", r.admission);
    println!("regen : rejeu rc={} en {:.1} s ; sortie {}", r.rc_enfant, r.duree_s, opts.sortie.display());
    let ecarts: Vec<_> = r.fichiers.iter().filter(|f| f.etat != "OK").collect();
    for f in ecarts.iter().take(10) {
        println!("DIFF {} : {} (attendu {}, obtenu {})", f.nom, f.etat, f.attendu.as_deref().unwrap_or("-"), f.obtenu.as_deref().unwrap_or("-"));
    }
    if ecarts.len() > 10 {
        println!("DIFF … {} autre(s) écart(s) (voir rapport_regen.json)", ecarts.len() - 10);
    }
    let verdict = if r.tout_identique() { "PASS" } else { "FAIL" };
    println!(
        "{verdict} REGEN {}/{} fichiers identiques à l'octet ; ancres {}/{} ; manifeste {}",
        r.identiques,
        r.fichiers.len(),
        r.ancres_identiques,
        r.ancres_comparees,
        r.manifeste_regen.display()
    );
    Ok(if r.tout_identique() { 0 } else { crate::commun::code::VALIDATION })
}

fn hachage(m: &ArgMatches) -> Hachage {
    if m.get_flag("hacher") {
        Hachage::Calculer
    } else {
        Hachage::Omettre
    }
}

fn adapter_un(man: &Path, vers: &Path, m: &ArgMatches) -> Resultat<i32> {
    if vers.exists() {
        return Err(Erreur::Config(format!("REFUS D'ÉCRASER : {} existe déjà", vers.display())));
    }
    if man.parent().is_some_and(|p| est_sous(vers, p)) {
        return Err(Erreur::Config("REFUS : la destination est sous le répertoire de l'ensemble d'origine".into()));
    }
    let mut ctx = contexte_avec_preuves(m)?;
    let v = adapter(man, &mut ctx, hachage(m))?;
    let erreurs = valider_manifeste(&v);
    if !erreurs.is_empty() {
        println!("FAIL REGEN-SCHEMA {} : {}", man.display(), erreurs.join(" ; "));
        return Ok(crate::commun::code::VALIDATION);
    }
    if let Some(p) = vers.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(p).map_err(|e| Erreur::io(format!("création de {}", p.display()), e))?;
    }
    std::fs::write(vers, serde_json::to_string_pretty(&v).map_err(|e| Erreur::Format(e.to_string()))? + "\n")
        .map_err(|e| Erreur::io(format!("écriture de {}", vers.display()), e))?;
    println!("PASS REGEN-SCHEMA {} classe {}", vers.display(), v["classe_regen"].as_str().unwrap_or("?"));
    Ok(0)
}

fn adapter_ensemble(ens: &Path, m: &ArgMatches) -> Resultat<i32> {
    let vers = PathBuf::from(
        m.get_one::<String>("vers").ok_or_else(|| Erreur::Config("--adapter-ensemble exige --vers DIR".into()))?,
    );
    if vers.exists() {
        return Err(Erreur::Config(format!("REFUS D'ÉCRASER : {} existe déjà", vers.display())));
    }
    if est_sous(&vers, ens) {
        return Err(Erreur::Config("REFUS : la destination est sous l'ensemble d'origine (aucune écriture dans un ensemble)".into()));
    }
    let mut ctx = contexte_avec_preuves(m)?;
    let chaines = trouver_chaines(ens)?;
    if chaines.is_empty() {
        return Err(Erreur::Config(format!("{} : aucune chaîne (aucun manifest.json)", ens.display())));
    }
    std::fs::create_dir_all(&vers).map_err(|e| Erreur::io(format!("création de {}", vers.display()), e))?;
    let (mut ok, mut ko) = (0usize, 0usize);
    let mut par_classe = [0usize; 3];
    for dir in chaines {
        let nom = dir.strip_prefix(ens).map(|p| p.to_string_lossy().replace('/', "_")).unwrap_or_default();
        let nom = if nom.is_empty() { "chaine".to_string() } else { nom };
        match adapter(&dir.join("manifest.json"), &mut ctx, hachage(m)) {
            Ok(v) => {
                let erreurs = valider_manifeste(&v);
                if erreurs.is_empty() {
                    let f = vers.join(format!("{nom}.json"));
                    std::fs::write(&f, serde_json::to_string_pretty(&v).map_err(|e| Erreur::Format(e.to_string()))? + "\n")
                        .map_err(|e| Erreur::io(format!("écriture de {}", f.display()), e))?;
                    ok += 1;
                    match v["classe_regen"].as_str() {
                        Some("A") => par_classe[0] += 1,
                        Some("B") => par_classe[1] += 1,
                        _ => par_classe[2] += 1,
                    }
                } else {
                    ko += 1;
                    println!("FAIL REGEN-SCHEMA {nom} : {}", erreurs.join(" ; "));
                }
            }
            Err(e) => {
                ko += 1;
                println!("FAIL REGEN-ADAPTATION {nom} : {e}");
            }
        }
    }
    println!(
        "{} REGEN-SCHEMA {ok} manifeste(s) kvu-regen/1 valides écrits dans {} (classes A {} / B {} / C {}), {ko} échec(s)",
        if ko == 0 { "PASS" } else { "FAIL" },
        vers.display(),
        par_classe[0],
        par_classe[1],
        par_classe[2]
    );
    Ok(if ko == 0 { 0 } else { crate::commun::code::VALIDATION })
}

/// `kvu store classer` : simulation.
pub fn executer_classer(m: &ArgMatches) -> Resultat<i32> {
    let mut ctx = contexte_avec_preuves(m)?;
    let racines: Vec<PathBuf> = m.get_many::<String>("chemin").map(|v| v.map(PathBuf::from).collect()).unwrap_or_default();
    let mut rapports = Vec::new();
    for r in &racines {
        if !r.is_dir() {
            return Err(Erreur::Config(format!("{} : ensemble introuvable", r.display())));
        }
        rapports.push(classer_ensemble(r, &mut ctx)?);
    }
    if m.get_flag("json") {
        let v = serde_json::json!({
            "simulation": true,
            "gbit_prouvees": ctx.gbit_vertes(),
            "ensembles": rapports.iter().map(|r| r.en_json()).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&v).map_err(|e| Erreur::Format(e.to_string()))?);
        return Ok(0);
    }
    println!("store classer : SIMULATION, rien n'est modifié (preuves G-BIT pour ce binaire : {:?})", ctx.gbit_vertes());
    for r in &rapports {
        let p = r.par_classe();
        println!(
            "ensemble {} : {} chaîne(s), {} fichier(s), {} ; classe dominante {}",
            r.racine.display(),
            r.chaines.len(),
            r.total_fichiers,
            taille_lisible(r.total_octets),
            r.classe_dominante()
        );
        for cl in [Classe::A, Classe::B, Classe::C] {
            let (n, o) = p[&cl];
            println!("  {cl} : {n} chaîne(s), {}", taille_lisible(o));
        }
        if r.hors_fichiers > 0 {
            println!("  hors chaînes (sans manifeste, classe C) : {} fichier(s), {}", r.hors_fichiers, taille_lisible(r.hors_octets));
        }
        if m.get_flag("detail") {
            for c in &r.chaines {
                println!("    {:<28} {}  {:>10}  {}", c.nom, c.classe, taille_lisible(c.octets), c.raisons.first().map(String::as_str).unwrap_or(""));
            }
        }
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn la_commande_regen_est_coherente() {
        commande_regen().debug_assert();
        commande_classer().debug_assert();
    }

    #[test]
    fn regen_sans_manifeste_est_un_usage_invalide() {
        let m = commande_regen().try_get_matches_from(["regen"]).unwrap();
        let e = executer_regen(&m).unwrap_err();
        assert_eq!(e.code_sortie(), crate::commun::code::USAGE);
    }

    #[test]
    fn adapter_ensemble_refuse_une_destination_sous_l_ensemble() {
        let d = std::env::temp_dir().join(format!("kvu_cliregen_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("c1")).unwrap();
        std::fs::write(d.join("c1/manifest.json"), b"{}").unwrap();
        let vers = d.join("manifestes");
        let m = commande_regen()
            .try_get_matches_from(["regen", "--adapter-ensemble", d.to_str().unwrap(), "--vers", vers.to_str().unwrap()])
            .unwrap();
        let e = executer_regen(&m).unwrap_err().to_string();
        assert!(e.contains("REFUS"), "{e}");
        assert!(!vers.exists());
        let _ = std::fs::remove_dir_all(&d);
    }
}
