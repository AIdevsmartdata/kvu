//! Ligne de commande unifiée (`kevinotron <sous-commande>` et binaire `kvu`), `valide`, `bench`.
//! Propriétaire : T9.
//!
//! Contrat des modules : chaque `<module>::cli` expose `commandes() -> Vec<clap::Command>` et
//! `executer(nom, &ArgMatches) -> Resultat<i32>` ; `principal` les assemble. Une erreur est
//! affichée en français et convertie en code de sortie stable (`commun::code`).
//!
//! `executer` reçoit les `ArgMatches` de la sous-commande elle-même (celles de la `Command`
//! rendue par `commandes()`), jamais ceux de la commande racine : `su2 gen …` appelle
//! `su2::cli::executer("su2", m)` avec `m.subcommand() == Some(("gen", …))`.
//!
//! Le dispatch est piloté par une table de modules ([`MODULES`]) ; la fonction
//! [`executer_argv`] prend la table et les flux de sortie en paramètres, ce qui la rend testable
//! avec des modules factices.

pub mod bench;
pub mod catalogue;
pub mod valide;

use crate::commun::{code, Erreur, Resultat};
use crate::gpu::sante::{est_faute_execution, Sante};
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Sous-commandes reconnues par le dispatch de `kevinotron` (main.rs les délègue ici avant clap).
/// Aucun drapeau historique ne commence sans `--` : pas d'ambiguïté avec l'ancienne interface.
pub const SOUS_COMMANDES: &[&str] = &[
    "doctor", "etat", "queue", "su2", "cond", "fibre", "obs", "store", "fetch", "valide", "bench", "regen", "groupe", "gen", "rg", "hess",
];

/// Version du crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Signature de `<module>::cli::commandes`.
pub type Declarateur = fn() -> Vec<clap::Command>;
/// Signature de `<module>::cli::executer`.
pub type Executeur = fn(&str, &clap::ArgMatches) -> Resultat<i32>;

/// Un module qui fournit des sous-commandes.
#[derive(Clone, Copy)]
pub struct Module {
    pub id: &'static str,
    pub commandes: Declarateur,
    pub executer: Executeur,
}

/// Les huit fournisseurs de sous-commandes (ordre d'affichage de l'aide).
pub const MODULES: &[Module] = &[
    Module {
        id: "machine",
        commandes: crate::machine::cli::commandes,
        executer: crate::machine::cli::executer,
    },
    Module {
        id: "su2",
        commandes: crate::su2::cli::commandes,
        executer: crate::su2::cli::executer,
    },
    Module {
        id: "cond",
        commandes: crate::cond::cli::commandes,
        executer: crate::cond::cli::executer,
    },
    Module {
        id: "fibre",
        commandes: crate::fibre::cli::commandes,
        executer: crate::fibre::cli::executer,
    },
    Module {
        id: "obs",
        commandes: crate::obs::cli::commandes,
        executer: crate::obs::cli::executer,
    },
    Module {
        id: "stockage",
        commandes: crate::stockage::cli::commandes,
        executer: crate::stockage::cli::executer,
    },
    Module {
        id: "regen",
        commandes: crate::regen::cli::commandes,
        executer: crate::regen::cli::executer,
    },
    Module {
        id: "groupe",
        commandes: crate::groupes::cli::commandes,
        executer: crate::groupes::cli::executer,
    },
    Module {
        id: "rg",
        commandes: crate::rg::cli::commandes,
        executer: crate::rg::cli::executer,
    },
    Module {
        id: "hess",
        commandes: crate::hess::cli::commandes,
        executer: crate::hess::cli::executer,
    },
    Module {
        id: "valide",
        commandes: valide::commandes,
        executer: valide::executer,
    },
    Module {
        id: "bench",
        commandes: bench::commandes,
        executer: bench::executer,
    },
];

fn hash_source() -> &'static str {
    // Variable injectée par crates/kvu/build.rs (T8) ; le préfixe historique reste accepté.
    option_env!("KEVINOTRON_SOURCE_HASH_UNIFIE")
        .or(option_env!("KEVINOTRON_SOURCE_HASH"))
        .unwrap_or("non_enregistre")
}

fn git_head() -> &'static str {
    option_env!("KEV_GIT_HEAD").unwrap_or("non_enregistre")
}

/// Texte de `--version` : version, source_hash de la recette courante et git_head embarqués.
pub fn version_texte() -> String {
    version_depuis(VERSION, hash_source(), git_head())
}

/// Version assemblée à partir de ses trois éléments et du nom de la recette (pure, testée).
pub fn version_depuis(version: &str, source_hash: &str, git: &str) -> String {
    let nom_recette = crate::stockage::recette::NOM_RECETTE;
    format!("{version}\nsource_hash ({nom_recette}) : {source_hash}\ngit_head : {git}")
}

/// Assemble la commande racine `kevinotron` avec toutes les sous-commandes des modules.
pub fn construire_commande(modules: &[Module]) -> clap::Command {
    let mut cmd = clap::Command::new("kvu")
        .bin_name("kvu")
        .about("kvu : théories de jauge sur réseau, machine reconnue à l'exécution (GPU HIP ou CUDA, sinon CPU)")
        .version(&*Box::leak(version_texte().into_boxed_str()))
        .subcommand_required(true)
        .arg_required_else_help(true);
    for m in modules {
        for sous in (m.commandes)() {
            cmd = cmd.subcommand(sous);
        }
    }
    cmd
}

/// Nom → module qui l'exécute, ou `None`. Un nom présent dans deux modules est une erreur de câblage
/// (détectée par le test `chaque_sous_commande_a_un_seul_proprietaire`).
pub fn proprietaire<'a>(modules: &'a [Module], nom: &str) -> Option<&'a Module> {
    modules
        .iter()
        .find(|m| (m.commandes)().iter().any(|c| c.get_name() == nom))
}

/// Point d'entrée : `argv` complet (argv[0] = binaire). Rend le code de sortie.
pub fn principal(argv: Vec<String>) -> i32 {
    let (sortie, erreur) = (std::io::stdout(), std::io::stderr());
    executer_argv(&argv, MODULES, &mut sortie.lock(), &mut erreur.lock())
}

/// Conduite à tenir après une faute d'exécution du GPU (règle de fiabilité : rejeu au bit près avant tout usage).
const CONSEIL_FAUTE_GPU: &str = "les sorties écrites depuis le dernier point de reprise sont suspectes : rejouer depuis ce point \
                                 et comparer au bit près avant de les garder";

/// Cœur testable de [`principal`] : table de modules et flux explicites. La santé GPU est celle du processus
/// (`gpu::sante::globale`) : toute faute d'exécution qu'elle a consignée interdit le code de sortie 0.
pub fn executer_argv(argv: &[String], modules: &[Module], sortie: &mut dyn Write, erreur: &mut dyn Write) -> i32 {
    executer_argv_avec(argv, modules, sortie, erreur, crate::gpu::sante::globale())
}

/// [`executer_argv`] avec un registre de santé GPU explicite (essais : un registre par essai, sans toucher à celui du processus).
fn executer_argv_avec(
    argv: &[String],
    modules: &[Module],
    sortie: &mut dyn Write,
    erreur: &mut dyn Write,
    sante: &Sante,
) -> i32 {
    let cmd = construire_commande(modules);
    let matches = match cmd.try_get_matches_from(argv) {
        Ok(m) => m,
        Err(e) => return afficher_erreur_clap(&e, sortie, erreur),
    };
    let Some((nom, sous)) = matches.subcommand() else {
        // Cas impossible (`subcommand_required`), conservé par prudence.
        let _ = writeln!(erreur, "kvu : aucune sous-commande (voir `kvu --help`)");
        return code::USAGE;
    };
    let Some(module) = proprietaire(modules, nom) else {
        let _ = writeln!(
            erreur,
            "kvu : sous-commande « {nom} » sans module exécutant (erreur de câblage)"
        );
        return code::ECHEC;
    };
    // Configuration `kvu.toml` lisible ? Sinon rc 2 avec fichier, ligne et colonne, avant toute autre action.
    if let Err(e) = crate::config::verifier() {
        let _ = writeln!(erreur, "kvu {nom} : {e}");
        return e.code_sortie();
    }
    // Préparation commune AVANT le module et avant tout fil : surcharge de machine cohérente (rc 2), affinité
    // obligatoire de la politique (clé `cpus` du profil, ex. cœurs E 12-19) posée sur le processus lui-même (fils et enfants héritent).
    if let Err(e) = crate::machine::processus::preparer() {
        let _ = writeln!(erreur, "kvu {nom} : {e}");
        return e.code_sortie();
    }
    // Une injection de panne GPU mal écrite (KVU_INJECTION_FAUTE_GPU) refuse la commande AVANT de l'exécuter : rc 2, jamais
    // ignorée en silence (une faute de frappe ne doit pas faire croire qu'un essai de panne a réussi).
    if let Err(e) = sante.verifier_configuration() {
        let _ = writeln!(erreur, "kvu {nom} : {e}");
        return e.code_sortie();
    }
    // Un panic dans un module (GPU, noyau) est converti en échec propre : jamais de rc 101 muet.
    let issue = catch_unwind(AssertUnwindSafe(|| (module.executer)(nom, sous)));
    let (rc, faute_deja_signalee) = match issue {
        Ok(Ok(rc)) => (rc, false),
        Ok(Err(e)) => {
            let _ = writeln!(erreur, "kvu {nom} : {e}");
            // Seule une faute d'exécution rendue par la commande dispense de rappeler le registre : une autre erreur (appareil
            // absent, configuration…) ne doit pas masquer une faute GPU consignée par ailleurs.
            let faute = matches!(e, Erreur::Gpu(_)) && est_faute_execution(&e.to_string());
            if faute {
                let _ = writeln!(
                    erreur,
                    "kvu {nom} : faute d'exécution GPU (code de sortie {}) : {CONSEIL_FAUTE_GPU}",
                    e.code_sortie()
                );
            }
            (e.code_sortie(), faute)
        }
        Err(_) => {
            let _ = writeln!(
                erreur,
                "kvu {nom} : erreur interne (panique du module « {} »)",
                module.id
            );
            (code::ECHEC, false)
        }
    };
    controle_de_sortie(nom, rc, faute_deja_signalee, sante, erreur)
}

/// Contrôle de sortie GPU. Toute faute d'exécution du GPU consignée par le runtime pendant la commande (erreur d'un destructeur,
/// dernière synchronisation d'un appareil, erreur que la commande a avalée) interdit le code 0 : le message est écrit sur stderr
/// et le code devient 3. Un code déjà non nul est conservé ; la faute est alors rappelée, sauf si la commande a elle-même rendu une
/// faute d'exécution du GPU en erreur (`deja_signale`).
fn controle_de_sortie(nom: &str, rc: i32, deja_signale: bool, sante: &Sante, erreur: &mut dyn Write) -> i32 {
    let Some(fautes) = sante.bilan_sortie() else { return rc };
    let detail = fautes.join(" | ");
    if rc == 0 {
        let _ = writeln!(
            erreur,
            "kvu {nom} : FAUTE D'EXÉCUTION GPU relevée pendant la commande, qui rendait pourtant 0 : {detail}"
        );
        let _ = writeln!(
            erreur,
            "kvu {nom} : code de sortie forcé à {} (faute GPU) : {CONSEIL_FAUTE_GPU}",
            code::GPU_INDISPONIBLE
        );
        code::GPU_INDISPONIBLE
    } else {
        if !deja_signale {
            let _ = writeln!(
                erreur,
                "kvu {nom} : faute(s) GPU consignée(s) pendant la commande (code de sortie conservé : {rc}) : {detail}"
            );
        }
        rc
    }
}

fn afficher_erreur_clap(e: &clap::Error, sortie: &mut dyn Write, erreur: &mut dyn Write) -> i32 {
    use clap::error::ErrorKind;
    let texte = e.render().to_string();
    match e.kind() {
        ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => {
            let _ = write!(sortie, "{texte}");
            code::OK
        }
        ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand => {
            let _ = write!(erreur, "{texte}");
            code::USAGE
        }
        _ => {
            let _ = write!(erreur, "{texte}");
            let _ = writeln!(
                erreur,
                "usage invalide : `kvu --help` ou `kvu <sous-commande> --help`"
            );
            code::USAGE
        }
    }
}

/// Convertit un résultat de module en code de sortie, en écrivant l'erreur en français.
pub fn code_de(nom: &str, r: Resultat<i32>, erreur: &mut dyn Write) -> i32 {
    match r {
        Ok(rc) => rc,
        Err(e) => {
            let _ = writeln!(erreur, "kvu {nom} : {e}");
            e.code_sortie()
        }
    }
}

/// Erreur d'usage (rc 2) pour les sous-commandes de T9.
pub(crate) fn usage(m: impl Into<String>) -> Erreur {
    Erreur::Config(m.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    fn cmd_fact(nom: &'static str) -> clap::Command {
        clap::Command::new(nom)
            .arg(clap::Arg::new("rc").long("rc").value_parser(clap::value_parser!(i32)))
            .arg(clap::Arg::new("verrou").long("verrou").action(clap::ArgAction::SetTrue))
            .arg(
                clap::Arg::new("panique")
                    .long("panique")
                    .action(clap::ArgAction::SetTrue),
            )
    }

    fn fact_a() -> Vec<clap::Command> {
        vec![cmd_fact("alpha"), cmd_fact("beta")]
    }
    fn fact_b() -> Vec<clap::Command> {
        vec![cmd_fact("gamma")]
    }

    fn exec_fact(nom: &str, m: &clap::ArgMatches) -> Resultat<i32> {
        if m.get_flag("panique") {
            panic!("panique volontaire de test ({nom})");
        }
        if m.get_flag("verrou") {
            return Err(Erreur::Verrou("délai dépassé".into()));
        }
        Ok(m.get_one::<i32>("rc").copied().unwrap_or(0))
    }

    const FAUX: &[Module] = &[
        Module {
            id: "a",
            commandes: fact_a,
            executer: exec_fact,
        },
        Module {
            id: "b",
            commandes: fact_b,
            executer: exec_fact,
        },
    ];

    fn lancer(args: &str) -> (i32, String, String) {
        let (mut o, mut e) = (Vec::new(), Vec::new());
        let rc = executer_argv(&argv(args), FAUX, &mut o, &mut e);
        (rc, String::from_utf8(o).unwrap(), String::from_utf8(e).unwrap())
    }

    #[test]
    fn dispatch_par_nom_et_code_de_sortie() {
        assert_eq!(lancer("kevinotron alpha").0, 0);
        assert_eq!(lancer("kevinotron gamma --rc 5").0, 5);
        assert_eq!(lancer("kevinotron beta --rc 42").0, 42);
    }

    #[test]
    fn erreur_affichee_en_francais_avec_code_stable() {
        let (rc, _, e) = lancer("kevinotron alpha --verrou");
        assert_eq!(rc, code::VERROU);
        assert!(e.contains("kvu alpha : verrou GPU : délai dépassé"), "{e}");
    }

    #[test]
    fn une_panique_est_convertie_en_echec_propre() {
        let (rc, _, e) = lancer("kevinotron gamma --panique");
        assert_eq!(rc, code::ECHEC);
        assert!(e.contains("erreur interne"), "{e}");
    }

    #[test]
    fn sous_commande_inconnue_ou_absente_rend_2() {
        assert_eq!(lancer("kevinotron zeta").0, code::USAGE);
        let (rc, _, e) = lancer("kevinotron");
        assert_eq!(rc, code::USAGE);
        assert!(
            e.contains("Usage") || e.contains("USAGE") || e.contains("Commands"),
            "{e}"
        );
    }

    #[test]
    fn option_inconnue_rend_2_avec_indication() {
        let (rc, _, e) = lancer("kevinotron alpha --nimporte");
        assert_eq!(rc, code::USAGE);
        assert!(e.contains("usage invalide"), "{e}");
    }

    #[test]
    fn aide_et_version_rendent_0_sur_la_sortie_standard() {
        let (rc, o, _) = lancer("kevinotron --help");
        assert_eq!(rc, 0);
        for nom in ["alpha", "beta", "gamma"] {
            assert!(o.contains(nom), "aide sans {nom} : {o}");
        }
        let (rc, o, _) = lancer("kevinotron --version");
        assert_eq!(rc, 0);
        assert!(o.contains("source_hash (kvu-source/1)") && o.contains("git_head"), "{o}");
        assert!(o.contains(VERSION), "{o}");
    }

    #[test]
    fn version_depuis_contient_les_trois_elements() {
        let v = version_depuis("1.2.3", "abcd", "ef01");
        assert_eq!(v, "1.2.3\nsource_hash (kvu-source/1) : abcd\ngit_head : ef01");
    }

    #[test]
    fn chaque_sous_commande_a_un_seul_proprietaire() {
        // Table réelle : les onze noms gelés sont fournis chacun par exactement un module,
        // et aucun module n'expose un nom hors de la liste gelée.
        let mut vus: Vec<String> = Vec::new();
        for m in MODULES {
            for c in (m.commandes)() {
                let nom = c.get_name().to_string();
                assert!(
                    SOUS_COMMANDES.contains(&nom.as_str()),
                    "{nom} (module {}) hors SOUS_COMMANDES",
                    m.id
                );
                assert!(!vus.contains(&nom), "{nom} fourni deux fois");
                vus.push(nom);
            }
        }
        for s in SOUS_COMMANDES {
            assert!(vus.iter().any(|v| v == s), "sous-commande gelée {s} sans fournisseur");
            assert_eq!(
                MODULES
                    .iter()
                    .filter(|m| (m.commandes)().iter().any(|c| c.get_name() == *s))
                    .count(),
                1
            );
        }
    }

    #[test]
    fn la_commande_racine_est_coherente() {
        construire_commande(MODULES).debug_assert();
        construire_commande(FAUX).debug_assert();
    }

    #[test]
    fn aucun_drapeau_historique_ne_collisionne() {
        // Le dispatch de main.rs ne se déclenche que si argv[1] est un nom nu : tout drapeau
        // historique commence par « - ».
        assert!(SOUS_COMMANDES.iter().all(|s| !s.starts_with('-')));
    }

    #[test]
    fn code_de_convertit_les_erreurs() {
        let mut e = Vec::new();
        assert_eq!(code_de("x", Ok(3), &mut e), 3);
        let r: Resultat<i32> = Err(Erreur::NonImplemente("m"));
        assert_eq!(code_de("x", r, &mut e), code::NON_IMPLEMENTE);
        assert!(String::from_utf8(e).unwrap().contains("non implémenté"));
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Santé GPU à la sortie (lot gpu-rc) : un registre PAR essai (`Sante::nouvelle`), jamais celui du processus, que les autres
    // essais de ce fichier consultent en parallèle.
    // ---------------------------------------------------------------------------------------------------------------

    const FAUTE_700: &str = "hipStreamSynchronize: an illegal memory access was encountered (700)";

    /// Commande factice qui rend une erreur du runtime GPU de texte `--texte T`, ou 0 sans option.
    fn cmd_gpu(nom: &'static str) -> clap::Command {
        clap::Command::new(nom)
            .arg(clap::Arg::new("texte").long("texte").num_args(1))
            .arg(clap::Arg::new("rc").long("rc").value_parser(clap::value_parser!(i32)))
    }

    fn fact_gpu() -> Vec<clap::Command> {
        vec![cmd_gpu("delta")]
    }

    fn exec_gpu(_nom: &str, m: &clap::ArgMatches) -> Resultat<i32> {
        match m.get_one::<String>("texte") {
            Some(t) => Err(Erreur::Gpu(t.clone())),
            None => Ok(m.get_one::<i32>("rc").copied().unwrap_or(0)),
        }
    }

    const FAUX_GPU: &[Module] = &[Module {
        id: "g",
        commandes: fact_gpu,
        executer: exec_gpu,
    }];

    fn lancer_sante(modules: &[Module], args: &[&str], sante: &Sante) -> (i32, String) {
        let (mut o, mut e) = (Vec::new(), Vec::new());
        let argv: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let rc = executer_argv_avec(&argv, modules, &mut o, &mut e, sante);
        (rc, String::from_utf8(e).unwrap())
    }

    #[test]
    fn une_faute_gpu_consignee_interdit_le_code_0_et_s_ecrit_sur_stderr() {
        let sante = Sante::nouvelle(None);
        // la commande rend 0 : elle n'a rien vu ; le runtime, lui, a consigné une faute (destructeur, dernière synchronisation)
        sante.consigner_texte("fin d'appareil HIP", FAUTE_700);
        let (rc, e) = lancer_sante(FAUX, &["kevinotron", "alpha"], &sante);
        assert_eq!(rc, code::GPU_INDISPONIBLE, "{e}");
        assert!(e.contains("kvu alpha : FAUTE D'EXÉCUTION GPU relevée pendant la commande, qui rendait pourtant 0"), "{e}");
        assert!(e.contains("fin d'appareil HIP : hipStreamSynchronize: an illegal memory access was encountered (700)"), "{e}");
        assert!(e.contains("code de sortie forcé à 3"), "{e}");
        assert!(e.contains("rejouer depuis ce point"), "{e}");
    }

    #[test]
    fn sans_faute_gpu_le_code_de_la_commande_est_inchange_et_stderr_reste_vide() {
        let sante = Sante::nouvelle(None);
        for rc in [0, 5, 42] {
            let (code_rendu, e) = lancer_sante(FAUX, &["kevinotron", "beta", "--rc", &rc.to_string()], &sante);
            assert_eq!(code_rendu, rc);
            assert!(e.is_empty(), "stderr non vide : {e}");
        }
    }

    #[test]
    fn un_code_non_nul_est_conserve_et_la_faute_est_rappelee() {
        let sante = Sante::nouvelle(None);
        sante.consigner_texte("fin d'appareil CUDA", FAUTE_700);
        let (rc, e) = lancer_sante(FAUX, &["kevinotron", "gamma", "--rc", "5"], &sante);
        assert_eq!(rc, 5, "{e}");
        assert!(e.contains("code de sortie conservé : 5") && e.contains("fin d'appareil CUDA"), "{e}");
    }

    #[test]
    fn une_erreur_gpu_de_la_commande_donne_3_avec_un_message_clair() {
        let sante = Sante::nouvelle(None);
        let (rc, e) = lancer_sante(FAUX_GPU, &["kevinotron", "delta", "--texte", FAUTE_700], &sante);
        assert_eq!(rc, code::GPU_INDISPONIBLE, "{e}");
        assert!(e.contains("kvu delta : GPU : hipStreamSynchronize: an illegal memory access was encountered (700)"), "{e}");
        assert!(e.contains("kvu delta : faute d'exécution GPU (code de sortie 3)"), "{e}");
        // rien n'est rappelé deux fois : le registre est vide ici, la commande a rendu l'erreur elle-même
        assert!(!e.contains("consignée"), "{e}");
    }

    #[test]
    fn une_erreur_gpu_qui_n_est_pas_une_faute_d_execution_garde_le_code_3_sans_le_conseil_de_rejeu() {
        let sante = Sante::nouvelle(None);
        let (rc, e) =
            lancer_sante(FAUX_GPU, &["kevinotron", "delta", "--texte", "backend cuda non compilé : reconstruire kvu"], &sante);
        assert_eq!(rc, code::GPU_INDISPONIBLE, "{e}");
        assert!(e.contains("kvu delta : GPU : backend cuda non compilé"), "{e}");
        assert!(!e.contains("faute d'exécution GPU"), "{e}");
    }

    #[test]
    fn une_erreur_gpu_deja_rendue_par_la_commande_n_est_pas_rappelee_par_le_controle_de_sortie() {
        let sante = Sante::nouvelle(None);
        sante.consigner_texte("synchroniser", FAUTE_700);
        let (rc, e) = lancer_sante(FAUX_GPU, &["kevinotron", "delta", "--texte", FAUTE_700], &sante);
        assert_eq!(rc, code::GPU_INDISPONIBLE, "{e}");
        assert_eq!(e.matches("illegal memory access").count(), 1, "la faute est écrite une seule fois : {e}");
    }

    #[test]
    fn une_erreur_gpu_qui_n_est_pas_une_faute_n_efface_pas_une_faute_consignee() {
        // la commande échoue pour une autre raison (appareil absent) alors que le runtime avait consigné une vraie faute : les deux
        // se lisent sur stderr, et le code reste non nul
        let sante = Sante::nouvelle(None);
        sante.consigner_texte("fin d'appareil HIP", FAUTE_700);
        let (rc, e) = lancer_sante(FAUX_GPU, &["kevinotron", "delta", "--texte", "backend cuda non compilé : reconstruire kvu"], &sante);
        assert_eq!(rc, code::GPU_INDISPONIBLE, "{e}");
        assert!(e.contains("kvu delta : GPU : backend cuda non compilé"), "{e}");
        assert!(e.contains("faute(s) GPU consignée(s) pendant la commande"), "{e}");
        assert!(e.contains("fin d'appareil HIP : hipStreamSynchronize: an illegal memory access was encountered (700)"), "{e}");
    }

    #[test]
    fn l_injection_de_sortie_fait_echouer_une_commande_sans_gpu() {
        use crate::gpu::sante::{Injection, Point};
        let sante = Sante::nouvelle(Some(Injection { point: Point::Sortie, rang: 1 }));
        let (rc, e) = lancer_sante(FAUX, &["kevinotron", "alpha"], &sante);
        assert_eq!(rc, code::GPU_INDISPONIBLE, "{e}");
        assert!(e.contains("[injection KVU_INJECTION_FAUTE_GPU=sortie] contrôle de sortie"), "{e}");
        assert!(e.contains("an illegal memory access was encountered (700)"), "{e}");
        // une faute injectée, pas un état collant : le registre est vidé de son injection par la première sortie
        let (rc2, e2) = lancer_sante(FAUX, &["kevinotron", "alpha"], &sante);
        assert_eq!(rc2, 0, "{e2}");
    }

    #[test]
    fn une_injection_mal_ecrite_refuse_la_commande_en_usage_sans_l_executer() {
        let sante = Sante::depuis_texte(Some("synchronisation:zéro"));
        // `--rc 7` : si la commande était exécutée, on lirait 7
        let (rc, e) = lancer_sante(FAUX, &["kevinotron", "alpha", "--rc", "7"], &sante);
        assert_eq!(rc, code::USAGE, "{e}");
        assert!(e.contains("kvu alpha : configuration : KVU_INJECTION_FAUTE_GPU"), "{e}");
    }
}
