//! Configuration des machines (`kvu.toml`) : analyseur TOML minimal, profils, recherche du fichier.
//!
//! Priorité des sources : variable `KEVINOTRON_*` > fichier `kvu.toml` > défauts compilés du profil `inconnue`.
//! Recherche du fichier : `KEVINOTRON_CONFIG` (obligatoire s'il est posé) > `./kvu.toml` > `~/.config/kvu/kvu.toml` >
//! défauts compilés (profil `inconnue` seulement : aucune machine n'est alors reconnue).
//!
//! Format : voir `docs/MACHINES.md`. Un fichier malformé donne le code de sortie 2 avec ligne et colonne.

pub mod profil;
pub mod toml_min;

pub use profil::{developper_chemin, ProfilConfig};

use crate::commun::{Erreur, Machine, Resultat};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Valeur de la clé racine `schema` (facultative) du fichier.
pub const SCHEMA: &str = "kvu-config/1";

/// D'où vient la configuration.
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    /// Fichier lu.
    Fichier(PathBuf),
    /// Aucun fichier trouvé : défauts compilés du profil `inconnue`.
    Defauts,
}

/// Configuration résolue : un profil par machine déclarée dans le fichier.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub profils: Vec<(Machine, ProfilConfig)>,
    pub source: Source,
}

fn erreur_toml(origine: &str, ligne: usize, colonne: usize, message: &str) -> Erreur {
    Erreur::Config(format!("{origine}:{ligne}:{colonne} : {message}"))
}

impl Config {
    /// Configuration sans fichier : seul le profil `inconnue`, aux défauts compilés.
    pub fn defauts() -> Config {
        Config {
            profils: vec![(Machine::Inconnue, ProfilConfig::defauts_inconnue())],
            source: Source::Defauts,
        }
    }

    /// Analyse le texte d'un `kvu.toml`. `origine` (chemin) figure dans les messages d'erreur (`origine:ligne:colonne`).
    pub fn depuis_texte(texte: &str, origine: &str) -> Resultat<Config> {
        let doc = toml_min::analyser(texte).map_err(|e| erreur_toml(origine, e.ligne, e.colonne, &e.message))?;
        let mut profils: Vec<(Machine, ProfilConfig)> = Vec::new();
        for table in &doc.tables {
            if table.nom.is_empty() {
                for (cle, e) in &table.entrees {
                    if cle != "schema" {
                        return Err(erreur_toml(
                            origine,
                            e.ligne,
                            e.colonne,
                            &format!("clé racine inconnue « {cle} » (seule « schema » est admise hors des profils)"),
                        ));
                    }
                    match &e.valeur {
                        toml_min::Valeur::Chaine(s) if s == SCHEMA => {}
                        autre => {
                            return Err(erreur_toml(
                                origine,
                                e.ligne,
                                e.colonne,
                                &format!("schema : « {SCHEMA} » attendu, trouvé {autre:?}"),
                            ))
                        }
                    }
                }
                continue;
            }
            let machine = match table.nom.strip_prefix("profils.") {
                Some(nom) if !nom.contains('.') => nom.parse::<Machine>().ok(),
                _ => None,
            };
            let Some(machine) = machine else {
                return Err(erreur_toml(
                    origine,
                    table.ligne,
                    table.colonne,
                    &format!(
                        "table inconnue « [{}] » (tables admises : [profils.<{}>])",
                        table.nom,
                        Machine::noms_possibles()
                    ),
                ));
            };
            let p = ProfilConfig::depuis_table(table)
                .map_err(|e| erreur_toml(origine, e.ligne, e.colonne, &e.message))?;
            profils.push((machine, p));
        }
        Ok(Config {
            profils,
            source: Source::Fichier(PathBuf::from(origine)),
        })
    }

    /// Lit un fichier `kvu.toml`.
    pub fn depuis_fichier(chemin: &Path) -> Resultat<Config> {
        let texte = std::fs::read_to_string(chemin)
            .map_err(|e| Erreur::Config(format!("{} : lecture impossible ({e})", chemin.display())))?;
        Config::depuis_texte(&texte, &chemin.display().to_string())
    }

    /// Recherche du fichier : `KEVINOTRON_CONFIG` > `<cwd>/kvu.toml` > `<home>/.config/kvu/kvu.toml` > défauts.
    pub fn chercher(env: &dyn Fn(&str) -> Option<String>, cwd: &Path, home: &Path) -> Resultat<Config> {
        if let Some(c) = env("KEVINOTRON_CONFIG").filter(|c| !c.trim().is_empty()) {
            // Un fichier désigné explicitement qui manque est une erreur (jamais de repli silencieux).
            return Config::depuis_fichier(Path::new(c.trim()));
        }
        for candidat in [cwd.join("kvu.toml"), home.join(".config/kvu/kvu.toml")] {
            if candidat.is_file() {
                return Config::depuis_fichier(&candidat);
            }
        }
        Ok(Config::defauts())
    }

    /// Profil de la machine `m` ; une machine absente du fichier prend les défauts compilés de `inconnue`.
    pub fn profil(&self, m: Machine) -> ProfilConfig {
        self.profils
            .iter()
            .find(|(x, _)| *x == m)
            .map(|(_, p)| p.clone())
            .unwrap_or_else(ProfilConfig::defauts_inconnue)
    }

    /// Machines reconnaissables (profils avec motifs), dans l'ordre de reconnaissance.
    pub fn machines_reconnaissables(&self) -> Vec<Machine> {
        Machine::TOUS
            .iter()
            .copied()
            .filter(|m| self.profils.iter().any(|(x, p)| x == m && p.reconnaissable()))
            .collect()
    }

    /// Reconnaît la machine d'après l'hôte, le modèle de CPU et les architectures de GPU visibles.
    pub fn identifier(&self, hote: &str, cpu: &str, archs: &[&str]) -> Machine {
        self.machines_reconnaissables()
            .into_iter()
            .find(|m| {
                self.profils
                    .iter()
                    .any(|(x, p)| x == m && p.correspond(hote, cpu, archs))
            })
            .unwrap_or(Machine::Inconnue)
    }
}

/// Architecture GPU imposée par le profil actif (clés `arch_hip`, `arch_cuda` de `kvu.toml`), ou None : l'architecture
/// est alors LUE SUR LE PÉRIPHÉRIQUE (HIP : `gcnArchName` ; CUDA : `sm_<majeur><mineur>` de la capacité de calcul).
pub fn arch_surcharge(backend: crate::commun::BackendKind) -> Option<String> {
    use crate::commun::BackendKind;
    let machine = crate::machine::detection::detecter_rapide().machine;
    let p = actuelle().profil(machine);
    let valeur = match backend {
        BackendKind::Hip => p.arch_hip,
        BackendKind::Cuda => p.arch_cuda,
        BackendKind::Cpu => String::new(),
    };
    ProfilConfig::texte_optionnel(&valeur)
}

static ACTUELLE: OnceLock<(Config, Option<String>)> = OnceLock::new();

fn charger() -> (Config, Option<String>) {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let cwd = std::env::current_dir().unwrap_or_default();
    match Config::chercher(&|k| std::env::var(k).ok(), &cwd, &home) {
        Ok(c) => (c, None),
        Err(e) => (Config::defauts(), Some(e.to_string())),
    }
}

/// Configuration du processus, chargée une fois. Si le chargement a échoué, ce sont les défauts compilés ;
/// l'erreur est rendue par [`verifier`] (la ligne de commande s'arrête alors avec le code 2).
pub fn actuelle() -> &'static Config {
    &ACTUELLE.get_or_init(charger).0
}

/// Vérifie que le fichier de configuration a pu être lu : sinon erreur `Config` (code de sortie 2).
pub fn verifier() -> Resultat<()> {
    match &ACTUELLE.get_or_init(charger).1 {
        None => Ok(()),
        Some(m) => Err(Erreur::Config(m.trim_start_matches("configuration : ").to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pas_d_env(_: &str) -> Option<String> {
        None
    }

    fn dossier(nom: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("kvu_config_{nom}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    const MINI: &str = "schema = \"kvu-config/1\"\n[profils.gamer]\nhotes = [\"h1\", \"h2\"]\ncpu = [\"i5\"]\ngpu_arch = [\"sm_99\"]\nssd_min_go = 12.5\nnice = 3\ncpus = \"2-4\"\ncable = \"c\"\n";

    fn kvu_toml_du_depot() -> Config {
        // Profils des deux machines de référence : fixture du crate (crates/kvu/kvu.toml). Le kvu.toml de la racine du
        // dépôt est un modèle neutre (profils `inconnue` et `location` seulement), testé par `kvu_toml_modele_neutre`.
        let chemin = Path::new(env!("CARGO_MANIFEST_DIR")).join("kvu.toml");
        Config::depuis_fichier(&chemin).expect("kvu.toml de référence du crate")
    }

    #[test]
    fn defauts_sont_le_profil_inconnue_prudent() {
        let c = Config::defauts();
        let p = c.profil(Machine::Inconnue);
        assert_eq!((p.ssd_min_go, p.ram_min_go, p.mem_max.as_str()), (20.0, 8.0, "8G"));
        assert_eq!(p.backend, "auto");
        assert!(!p.reconnaissable());
        assert_eq!(c.source, Source::Defauts);
        // aucune machine n'est reconnue sans fichier
        assert_eq!(c.identifier("h1", "i5", &["sm_99"]), Machine::Inconnue);
        assert!(c.machines_reconnaissables().is_empty());
    }

    #[test]
    fn profil_absent_du_fichier_prend_les_defauts_inconnue() {
        let c = Config::depuis_texte(MINI, "x.toml").unwrap();
        assert_eq!(c.profil(Machine::Acemagic), ProfilConfig::defauts_inconnue());
        assert_eq!(c.profil(Machine::Location), ProfilConfig::defauts_inconnue());
    }

    #[test]
    fn valeurs_du_fichier_et_cles_absentes_aux_defauts() {
        let c = Config::depuis_texte(MINI, "x.toml").unwrap();
        let p = c.profil(Machine::Gamer);
        assert_eq!((p.ssd_min_go, p.nice, p.cpus.as_str(), p.cable.as_str()), (12.5, 3, "2-4", "c"));
        let d = ProfilConfig::defauts_inconnue();
        assert_eq!((p.ram_min_go, p.mem_max.clone(), p.verrou.clone()), (d.ram_min_go, d.mem_max, d.verrou));
    }

    #[test]
    fn identification_par_hote_cpu_ou_gpu() {
        let c = Config::depuis_texte(MINI, "x.toml").unwrap();
        assert_eq!(c.identifier(" h2 ", "", &[]), Machine::Gamer);
        assert_eq!(c.identifier("autre", "Intel Core I5-14600KF", &[]), Machine::Gamer);
        assert_eq!(c.identifier("autre", "", &["sm_70", "sm_99"]), Machine::Gamer);
        assert_eq!(c.identifier("autre", "amd", &["gfx1"]), Machine::Inconnue);
    }

    #[test]
    fn kvu_toml_du_depot_reproduit_les_valeurs_historiques() {
        // Valeurs des défauts compilés de 0a7a2de (ARCHITECTURE_UNIFIEE.md §3), recopiées ici comme témoin.
        let c = kvu_toml_du_depot();
        let a = c.profil(Machine::Acemagic);
        assert_eq!(
            (a.verrou.as_str(), a.nice, a.cpus.as_str(), a.fils_max, a.mem_max.as_str(), a.ram_min_go, a.ssd_min_go),
            ("/tmp/kevinotron_gpu.lock", 10, "", 8, "24G", 25.0, 100.0)
        );
        assert_eq!((a.precision.as_str(), a.cable.as_str(), a.backend.as_str()), ("f32", "gamer-eth", "hip"));
        let g = c.profil(Machine::Gamer);
        assert_eq!(
            (g.verrou.as_str(), g.nice, g.cpus.as_str(), g.fils_max, g.mem_max.as_str(), g.ram_min_go, g.ssd_min_go),
            ("/tmp/gamer_gpu.lock", 5, "12-19", 8, "12G", 6.0, 50.0)
        );
        assert_eq!((g.precision.as_str(), g.cable.as_str(), g.alerte.as_str(), g.backend.as_str()), ("f32", "", "", "cuda"));
        let i = c.profil(Machine::Inconnue);
        assert_eq!(i, ProfilConfig::defauts_inconnue(), "le profil inconnue du fichier = les défauts compilés");
        assert_eq!(c.machines_reconnaissables(), vec![Machine::Acemagic, Machine::Gamer]);
    }

    #[test]
    fn profil_location_sans_nom_d_hote_ni_cable_ni_ssh() {
        let l = kvu_toml_du_depot().profil(Machine::Location);
        assert!(l.hotes.is_empty() && l.cpu.is_empty() && l.gpu_arch.is_empty());
        assert!(l.cable.is_empty(), "pas de câble, donc pas de ssh");
        assert_eq!(l.verrou, "/tmp/kvu_gpu.lock");
        assert!(l.store.starts_with("~/"), "store sous $HOME : {}", l.store);
        assert!(l.arch_hip.is_empty() && l.arch_cuda.is_empty(), "architecture lue sur le périphérique");
    }

    #[test]
    fn kvu_toml_modele_neutre() {
        // Le kvu.toml de la racine du dépôt est un modèle : il ne reconnaît aucune machine (ni par l'hôte, ni par le CPU,
        // ni par le GPU), ses profils inconnue et location sont ceux de la fixture, et il ne contient aucun chemin absolu
        // hors de /tmp.
        let chemin = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kvu.toml");
        let m = Config::depuis_fichier(&chemin).expect("kvu.toml modèle de la racine");
        assert!(m.machines_reconnaissables().is_empty());
        for (hote, cpu, arch) in [
            ("acemagic", "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S", "gfx1151"),
            ("pc", "Intel(R) Core(TM) i5-14600KF", "sm_120"),
        ] {
            assert_eq!(m.identifier(hote, cpu, &[arch]), Machine::Inconnue, "{hote}");
        }
        let f = kvu_toml_du_depot();
        assert_eq!(m.profil(Machine::Inconnue), f.profil(Machine::Inconnue));
        assert_eq!(m.profil(Machine::Location), f.profil(Machine::Location));
        let texte = std::fs::read_to_string(&chemin).unwrap();
        for l in texte.lines().filter(|l| !l.trim_start().starts_with('#')) {
            if let Some((_, v)) = l.split_once('=') {
                let v = v.trim().trim_matches('"');
                assert!(!v.starts_with('/') || v.starts_with("/tmp/"), "chemin absolu dans le modèle : {l}");
            }
        }
    }

    fn rejet(texte: &str) -> String {
        match Config::depuis_texte(texte, "t.toml") {
            Err(Erreur::Config(m)) => m,
            autre => panic!("rejet attendu, obtenu {autre:?}"),
        }
    }

    #[test]
    fn rejet_table_inconnue_avec_ligne_et_colonne() {
        let m = rejet("schema = \"kvu-config/1\"\n\n[profils.cray]\nnice = 1\n");
        assert!(m.starts_with("t.toml:3:1 :") && m.contains("table inconnue"), "{m}");
        let m = rejet("[autre]\n");
        assert!(m.starts_with("t.toml:1:1 :") && m.contains("table inconnue"), "{m}");
        let m = rejet("[profils.gamer.sous]\n");
        assert!(m.contains("table inconnue"), "{m}");
    }

    #[test]
    fn rejet_cle_inconnue_type_et_valeurs() {
        let m = rejet("[profils.gamer]\nnice = 1\nbidule = 2\n");
        assert!(m.starts_with("t.toml:3:1 :") && m.contains("clé inconnue"), "{m}");
        let m = rejet("[profils.gamer]\nssd_min_go = \"beaucoup\"\n");
        assert!(m.starts_with("t.toml:2:1 :") && m.contains("nombre attendu"), "{m}");
        assert!(rejet("[profils.gamer]\nssd_min_go = -1\n").contains("positif"));
        assert!(rejet("[profils.gamer]\nnice = 40\n").contains("entre -20 et 19"));
        assert!(rejet("[profils.gamer]\nbackend = \"vulkan\"\n").contains("auto|hip|cuda|cpu"));
        assert!(rejet("[profils.gamer]\nmem_max = \"lot\"\n").contains("24G"));
        assert!(rejet("[profils.gamer]\ncpus = \"a-b\"\n").contains("liste de cœurs"));
        assert!(rejet("[profils.gamer]\nprecision = \"f16\"\n").contains("f64|f32"));
        assert!(rejet("[profils.gamer]\nhotes = \"seul\"\n").contains("tableau de chaînes"));
        assert!(rejet("[profils.gamer]\nfils_max = 0\n").contains("au moins 1"));
        assert!(rejet("[profils.gamer]\nstore = \"\"\n").contains("non vide"));
    }

    #[test]
    fn rejet_schema_clef_racine_et_toml_malforme() {
        assert!(rejet("schema = \"autre/9\"\n").contains("schema"));
        assert!(rejet("nice = 1\n").contains("clé racine inconnue"));
        let m = rejet("[profils.gamer]\nnice = 1\nnice = 2\n");
        assert!(m.starts_with("t.toml:3:1 :") && m.contains("dupliquée"), "{m}");
        let m = rejet("[profils.gamer]\nstore = \"ouvert\n");
        assert!(m.starts_with("t.toml:2:9 :") && m.contains("non fermée"), "{m}");
        assert!(rejet("[profils.gamer]\n[profils.gamer]\n").contains("dupliqué"));
        // rc 2 (usage/configuration) pour toutes ces erreurs
        let e = Config::depuis_texte("[x\n", "t.toml").unwrap_err();
        assert_eq!(e.code_sortie(), 2);
    }

    #[test]
    fn recherche_variable_puis_cwd_puis_home_puis_defauts() {
        let d = dossier("recherche");
        let (cwd, home) = (d.join("cwd"), d.join("home"));
        std::fs::create_dir_all(home.join(".config/kvu")).unwrap();
        std::fs::create_dir_all(&cwd).unwrap();
        let ecrire = |p: &Path, ssd: u32| std::fs::write(p, format!("[profils.inconnue]\nssd_min_go = {ssd}\n")).unwrap();
        let (f_env, f_cwd, f_home) = (d.join("env.toml"), cwd.join("kvu.toml"), home.join(".config/kvu/kvu.toml"));
        let ssd = |c: Resultat<Config>| c.unwrap().profil(Machine::Inconnue).ssd_min_go;
        // rien : défauts compilés
        assert_eq!(ssd(Config::chercher(&pas_d_env, &cwd, &home)), 20.0);
        ecrire(&f_home, 3);
        assert_eq!(ssd(Config::chercher(&pas_d_env, &cwd, &home)), 3.0);
        ecrire(&f_cwd, 2);
        assert_eq!(ssd(Config::chercher(&pas_d_env, &cwd, &home)), 2.0);
        ecrire(&f_env, 1);
        let env = |k: &str| (k == "KEVINOTRON_CONFIG").then(|| f_env.display().to_string());
        assert_eq!(ssd(Config::chercher(&env, &cwd, &home)), 1.0);
        // un fichier désigné par la variable et absent : erreur, jamais de repli silencieux
        let absent = |k: &str| (k == "KEVINOTRON_CONFIG").then(|| d.join("nulle_part.toml").display().to_string());
        let e = Config::chercher(&absent, &cwd, &home).unwrap_err();
        assert_eq!(e.code_sortie(), 2);
        // variable vide : ignorée
        let vide = |k: &str| (k == "KEVINOTRON_CONFIG").then(|| "  ".to_string());
        assert_eq!(ssd(Config::chercher(&vide, &cwd, &home)), 2.0);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn chemins_tilde_developpes_sous_home() {
        let h = Path::new("/home/loc");
        assert_eq!(developper_chemin("~/x/y", h), PathBuf::from("/home/loc/x/y"));
        assert_eq!(developper_chemin("~", h), PathBuf::from("/home/loc"));
        assert_eq!(developper_chemin("/abs/p", h), PathBuf::from("/abs/p"));
        assert_eq!(ProfilConfig::chemin_optionnel("  ", h), None);
    }

    #[test]
    fn noms_de_profils_viennent_des_variantes() {
        for m in Machine::TOUS {
            assert_eq!(m.nom().parse::<Machine>().unwrap(), m);
            assert_eq!(m.nom(), m.nom().to_ascii_lowercase());
        }
        assert!(Machine::Inconnue.est_generique() && Machine::Location.est_generique());
        assert!(!Machine::Acemagic.est_generique() && !Machine::Gamer.est_generique());
        assert!("cray".parse::<Machine>().is_err());
    }
}
