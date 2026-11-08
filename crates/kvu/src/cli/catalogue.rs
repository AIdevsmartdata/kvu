//! Catalogue des gates attendues (ARCHITECTURE_UNIFIEE.md §6) et analyse de la sortie normalisée
//! de `validation/gates_unifie.sh`.
//!
//! Principe : une gate qui ne sort pas du tout ne doit pas passer inaperçue. Le catalogue liste,
//! pour chaque machine, les gates attendues ; `completer` ajoute une ligne FAIL « gate attendue
//! non émise » pour chacune qui manque. Un SKIP n'est jamais vert.
//!
//! Format normalisé des lignes de gates (émis par gates_unifie.sh, séparateur = tabulation) :
//! `GATE<TAB>PASS|FAIL|SKIP<TAB>machine<TAB>nom<TAB>détail`.

use serde::Serialize;

/// Mode de la suite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Gates CPU seulement : ni GPU, ni compilation lourde, ni autre machine.
    Rapide,
    /// Toutes les gates de §6, sur les deux machines.
    Complet,
}

impl Mode {
    pub fn nom(self) -> &'static str {
        match self {
            Mode::Rapide => "rapide",
            Mode::Complet => "complet",
        }
    }
}

/// Statut d'une gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Statut {
    #[serde(rename = "PASS")]
    Pass,
    #[serde(rename = "FAIL")]
    Fail,
    #[serde(rename = "SKIP")]
    Skip,
    /// Ligne rapportée, ni PASS ni FAIL (arbitrage D3 : OBS-4-L8). Ne bloque pas, ne verdit pas.
    #[serde(rename = "INFO")]
    Info,
    /// Rouge arbitré et documenté (arbitrage D3 : G-H11, fait statistique). Non bloquant, toujours listé ;
    /// n'est PAS un PASS.
    #[serde(rename = "FAIL-DOCUMENTE")]
    FailDocumente,
}

impl Statut {
    pub fn analyser(s: &str) -> Option<Statut> {
        match s {
            "PASS" => Some(Statut::Pass),
            "FAIL" => Some(Statut::Fail),
            "SKIP" => Some(Statut::Skip),
            "INFO" => Some(Statut::Info),
            "FAIL-DOCUMENTE" => Some(Statut::FailDocumente),
            _ => None,
        }
    }
    pub fn nom(self) -> &'static str {
        match self {
            Statut::Pass => "PASS",
            Statut::Fail => "FAIL",
            Statut::Skip => "SKIP",
            Statut::Info => "INFO",
            Statut::FailDocumente => "FAIL-DOCUMENTE",
        }
    }
}

/// Une gate rendue.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Gate {
    pub nom: String,
    pub machine: String,
    pub statut: Statut,
    pub detail: String,
}

/// Une gate attendue. `jeton` peut lister des alias séparés par `|` ; une gate émise la couvre si son
/// nom CONTIENT un des alias (insensible à la casse) : « BORD-C1/C2a » couvre « BORD-C1 » et « BORD-C2a ».
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Attendue {
    pub jeton: &'static str,
    /// Machines sur lesquelles elle doit être émise.
    pub machines: &'static [&'static str],
    /// Dans le mode rapide ?
    pub rapide: bool,
    /// Tâche propriétaire (information).
    pub tache: &'static str,
}

const AM: &[&str] = &[crate::commun::NOM_PROFIL_ACEMAGIC];
const GA: &[&str] = &[crate::commun::NOM_PROFIL_GAMER];
const DEUX: &[&str] = &[crate::commun::NOM_PROFIL_ACEMAGIC, crate::commun::NOM_PROFIL_GAMER];

/// Catalogue complet de §6.1 à §6.6 (les 13 gates historiques comprises).
pub const CATALOGUE: &[Attendue] = &[
    // §6.1 bit-identité et déterminisme (T3)
    Attendue {
        jeton: "G-BIT-K1",
        machines: AM,
        rapide: false,
        tache: "T3",
    },
    Attendue {
        jeton: "G-BIT-HIP-PROD",
        machines: AM,
        rapide: false,
        tache: "T3",
    },
    Attendue {
        jeton: "G-BIT-CUDA",
        machines: GA,
        rapide: false,
        tache: "T3",
    },
    Attendue {
        jeton: "G-BIT-BORD",
        machines: GA,
        rapide: false,
        tache: "T3",
    },
    Attendue {
        jeton: "G-CPU-GPU",
        machines: DEUX,
        rapide: false,
        tache: "T3",
    },
    Attendue {
        jeton: "G-DET",
        machines: DEUX,
        rapide: false,
        tache: "T3",
    },
    Attendue {
        jeton: "LOT-1",
        machines: DEUX,
        rapide: false,
        tache: "T3",
    },
    Attendue {
        jeton: "REPRISE-1",
        machines: DEUX,
        rapide: false,
        tache: "T3",
    },
    Attendue {
        jeton: "BORD-C1",
        machines: DEUX,
        rapide: false,
        tache: "T3",
    },
    Attendue {
        jeton: "BORD-C2a|BORD-C1/C2a",
        machines: DEUX,
        rapide: false,
        tache: "T3",
    },
    // §6.2 loi du chemin FP32 (T4)
    Attendue {
        jeton: "LOI-0",
        machines: DEUX,
        rapide: false,
        tache: "T4",
    },
    Attendue {
        jeton: "LOI-1",
        machines: DEUX,
        rapide: false,
        tache: "T4",
    },
    Attendue {
        jeton: "LOI-2",
        machines: DEUX,
        rapide: false,
        tache: "T4",
    },
    Attendue {
        jeton: "LOI-3",
        machines: DEUX,
        rapide: false,
        tache: "T4",
    },
    Attendue {
        jeton: "LOI-4",
        machines: DEUX,
        rapide: false,
        tache: "T4",
    },
    Attendue {
        jeton: "LOI-τ|LOI-tau",
        machines: DEUX,
        rapide: false,
        tache: "T4",
    },
    // §6.3 bain conditionnel (T5) et fibre (T6)
    Attendue {
        jeton: "RNG-NP",
        machines: DEUX,
        rapide: false,
        tache: "T5",
    },
    Attendue {
        jeton: "EQC-1",
        machines: DEUX,
        rapide: false,
        tache: "T5",
    },
    Attendue {
        jeton: "EQC-1b",
        machines: DEUX,
        rapide: false,
        tache: "T5",
    },
    Attendue {
        jeton: "EQC-2",
        machines: DEUX,
        rapide: false,
        tache: "T5",
    },
    Attendue {
        jeton: "EQC-3",
        machines: DEUX,
        rapide: false,
        tache: "T5",
    },
    Attendue {
        jeton: "EQC-4",
        machines: DEUX,
        rapide: false,
        tache: "T5",
    },
    Attendue {
        jeton: "FIB-1",
        machines: DEUX,
        rapide: false,
        tache: "T6",
    },
    Attendue {
        jeton: "FIB-2",
        machines: DEUX,
        rapide: false,
        tache: "T6",
    },
    // §6.4 observables (T7)
    Attendue {
        jeton: "OBS-1",
        machines: DEUX,
        rapide: false,
        tache: "T7",
    },
    Attendue {
        jeton: "OBS-2",
        machines: DEUX,
        rapide: false,
        tache: "T7",
    },
    Attendue {
        jeton: "OBS-3",
        machines: DEUX,
        rapide: false,
        tache: "T7",
    },
    // Arbitrage D3 (GATES v1, WP2) : OBS-4 (L = 8, rapportée) devient INFO ; la gate de quantification est OBS-4-L16.
    Attendue {
        jeton: "OBS-4-L8",
        machines: DEUX,
        rapide: false,
        tache: "T7",
    },
    Attendue {
        jeton: "OBS-4-L16",
        machines: DEUX,
        rapide: false,
        tache: "T7",
    },
    // Mutant d'OBS-1 (+1e-14 sur un W de la référence CPU) : doit être détecté ROUGE (GATES v1 règle 4). Nom sans
    // tiret entre OBS et 1 pour n'être couvert que par son propre jeton.
    Attendue {
        jeton: "OBS1-MUTANT",
        machines: DEUX,
        rapide: false,
        tache: "T7",
    },
    // §6.5 infrastructure (T1, T2, T8)
    Attendue {
        jeton: "RT-1",
        machines: DEUX,
        rapide: false,
        tache: "T2",
    },
    Attendue {
        jeton: "RT-2",
        machines: DEUX,
        rapide: false,
        tache: "T2",
    },
    Attendue {
        jeton: "MACH-1",
        machines: DEUX,
        rapide: false,
        tache: "T1",
    },
    Attendue {
        jeton: "VERROU-1",
        machines: DEUX,
        rapide: false,
        tache: "T1",
    },
    Attendue {
        jeton: "FILE-1",
        machines: DEUX,
        rapide: false,
        tache: "T1",
    },
    Attendue {
        jeton: "PROV-1",
        machines: AM,
        rapide: false,
        tache: "T8",
    },
    Attendue {
        jeton: "PROV-2",
        machines: AM,
        rapide: false,
        tache: "T8",
    },
    Attendue {
        jeton: "PROV-3",
        machines: AM,
        rapide: false,
        tache: "T8",
    },
    Attendue {
        jeton: "SSD-1",
        machines: AM,
        rapide: false,
        tache: "T8",
    },
    Attendue {
        jeton: "SSD-2",
        machines: AM,
        rapide: false,
        tache: "T8",
    },
    Attendue {
        jeton: "FETCH-1",
        machines: AM,
        rapide: false,
        tache: "T8",
    },
    // §6.6 les 13 gates historiques (T9)
    Attendue {
        jeton: "G-H01",
        machines: DEUX,
        rapide: true,
        tache: "T9",
    },
    Attendue {
        jeton: "G-H02",
        machines: DEUX,
        rapide: true,
        tache: "T9",
    },
    Attendue {
        jeton: "G-H03",
        machines: DEUX,
        rapide: false,
        tache: "T9",
    },
    Attendue {
        jeton: "G-H04",
        machines: DEUX,
        rapide: false,
        tache: "T9",
    },
    Attendue {
        jeton: "G-H05",
        machines: DEUX,
        rapide: false,
        tache: "T9",
    },
    Attendue {
        jeton: "G-H06",
        machines: GA,
        rapide: false,
        tache: "T9",
    },
    Attendue {
        jeton: "G-H07",
        machines: AM,
        rapide: false,
        tache: "T9",
    },
    Attendue {
        jeton: "G-H08",
        machines: AM,
        rapide: false,
        tache: "T9",
    },
    Attendue {
        jeton: "G-H09",
        machines: AM,
        rapide: false,
        tache: "T9",
    },
    Attendue {
        jeton: "G-H10",
        machines: AM,
        rapide: false,
        tache: "T9",
    },
    Attendue {
        jeton: "G-H11",
        machines: AM,
        rapide: false,
        tache: "T9",
    },
    Attendue {
        jeton: "G-H12",
        machines: AM,
        rapide: false,
        tache: "T9",
    },
    Attendue {
        jeton: "G-H13",
        machines: AM,
        rapide: false,
        tache: "T9",
    },
    Attendue {
        jeton: "TESTS-CPU",
        machines: DEUX,
        rapide: true,
        tache: "T9",
    },
];

impl Attendue {
    /// Vrai si une gate émise de nom `nom` couvre ce jeton.
    pub fn couverte_par(&self, nom: &str) -> bool {
        let n = nom.to_lowercase();
        self.jeton.split('|').any(|alias| couvre(&n, &alias.to_lowercase()))
    }
}

/// `nom` contient `alias` comme un jeton entier : le caractère qui suit n'est pas un chiffre ni
/// une lettre, sauf pour les jetons terminés par un chiffre qui acceptent une suite alphabétique
/// courte (« EQC-1 » ne couvre pas « EQC-10 » mais « EQC-1b » est un jeton propre du catalogue).
fn couvre(nom: &str, alias: &str) -> bool {
    let mut debut = 0;
    while let Some(p) = nom[debut..].find(alias) {
        let fin = debut + p + alias.len();
        let suite = nom[fin..].chars().next();
        let prec = nom[..debut + p].chars().last();
        let bord_gauche = prec.is_none_or(|c| !c.is_alphanumeric());
        let bord_droit = match suite {
            None => true,
            Some(c) => !c.is_alphanumeric(),
        };
        if bord_gauche && bord_droit {
            return true;
        }
        debut = debut + p + 1;
        if debut >= nom.len() {
            break;
        }
    }
    false
}

/// Analyse une ligne normalisée `GATE\tSTATUT\tmachine\tnom\tdétail`. Les autres lignes sont ignorées.
pub fn analyser_ligne(ligne: &str) -> Option<Gate> {
    let mut it = ligne.trim_end_matches(['\r', '\n']).splitn(5, '\t');
    if it.next()? != "GATE" {
        return None;
    }
    let statut = Statut::analyser(it.next()?)?;
    let machine = it.next()?.trim().to_string();
    let nom = it.next()?.trim().to_string();
    if machine.is_empty() || nom.is_empty() {
        return None;
    }
    let detail = it.next().unwrap_or("").trim().to_string();
    Some(Gate {
        nom,
        machine,
        statut,
        detail,
    })
}

/// Analyse toute la sortie du script.
pub fn analyser_sortie(texte: &str) -> Vec<Gate> {
    texte.lines().filter_map(analyser_ligne).collect()
}

/// Filtre un catalogue selon le mode, les machines demandées et un éventuel motif de gate.
pub fn attendues(mode: Mode, machines: &[String], motif: Option<&str>) -> Vec<(Attendue, &'static str)> {
    let mut v = Vec::new();
    for a in CATALOGUE {
        if mode == Mode::Rapide && !a.rapide {
            continue;
        }
        if let Some(m) = motif {
            if !a.jeton.to_lowercase().contains(&m.to_lowercase()) {
                continue;
            }
        }
        for &mach in a.machines {
            if machines.iter().any(|x| x == mach) {
                v.push((*a, mach));
            }
        }
    }
    v
}

/// Ajoute un FAIL pour chaque gate attendue qui n'a pas été émise (nom et machine). Rend la liste
/// complète (émises, puis manquantes).
pub fn completer(mut emises: Vec<Gate>, attendues: &[(Attendue, &'static str)]) -> Vec<Gate> {
    for (a, mach) in attendues {
        let presente = emises.iter().any(|g| g.machine == *mach && a.couverte_par(&g.nom));
        if !presente {
            emises.push(Gate {
                nom: a.jeton.split('|').next().unwrap_or(a.jeton).to_string(),
                machine: (*mach).to_string(),
                statut: Statut::Fail,
                detail: format!("gate attendue (tâche {}) non émise par la suite", a.tache),
            });
        }
    }
    emises
}

/// Totaux d'une liste de gates : (PASS, FAIL, SKIP). Les lignes INFO et FAIL-DOCUMENTE n'y figurent pas.
pub fn totaux(gates: &[Gate]) -> (usize, usize, usize) {
    let n = |s| gates.iter().filter(|g| g.statut == s).count();
    (n(Statut::Pass), n(Statut::Fail), n(Statut::Skip))
}

/// Lignes arbitrées par la décision D3 : (INFO, FAIL-DOCUMENTE). Jamais comptées comme PASS.
pub fn totaux_arbitres(gates: &[Gate]) -> (usize, usize) {
    let n = |s| gates.iter().filter(|g| g.statut == s).count();
    (n(Statut::Info), n(Statut::FailDocumente))
}

/// Sortie JSON gelée : `{"gates":[{"nom","machine","statut","detail"}],"fail":n,"skip":n}`.
pub fn json(gates: &[Gate]) -> String {
    #[derive(Serialize)]
    struct Sortie<'a> {
        gates: &'a [Gate],
        fail: usize,
        skip: usize,
        info: usize,
        fail_documente: usize,
    }
    let (_, fail, skip) = totaux(gates);
    let (info, fail_documente) = totaux_arbitres(gates);
    serde_json::to_string(&Sortie {
        gates,
        fail,
        skip,
        info,
        fail_documente,
    })
    .expect("sérialisation JSON infaillible")
}

/// Tableau lisible (machine, gate, statut, détail).
pub fn tableau(gates: &[Gate]) -> String {
    let mut s = String::new();
    s.push_str(&format!("{:<9} {:<18} {:<5} {}\n", "machine", "gate", "état", "détail"));
    for g in gates {
        s.push_str(&format!(
            "{:<9} {:<18} {:<5} {}\n",
            g.machine,
            g.nom,
            g.statut.nom(),
            g.detail
        ));
    }
    let (p, f, k) = totaux(gates);
    let (i, d) = totaux_arbitres(gates);
    s.push_str(&format!(
        "TOTAL : {p} PASS, {f} FAIL, {k} SKIP, {i} INFO, {d} FAIL-DOCUMENTE (un SKIP n'est pas vert ; INFO et FAIL-DOCUMENTE sont arbitrés par D3 et listés)\n"
    ));
    s
}

/// Code de sortie : 0 seulement si ni FAIL ni SKIP (INFO et FAIL-DOCUMENTE, arbitrés par D3, ne bloquent pas).
pub fn code_de(gates: &[Gate]) -> i32 {
    let (_, f, k) = totaux(gates);
    if f == 0 && k == 0 && !gates.is_empty() {
        crate::commun::code::OK
    } else {
        crate::commun::code::VALIDATION
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(nom: &str, machine: &str, s: Statut) -> Gate {
        Gate {
            nom: nom.into(),
            machine: machine.into(),
            statut: s,
            detail: String::new(),
        }
    }

    #[test]
    fn ligne_normalisee_analysee() {
        let l = "GATE\tPASS\tacemagic\tG-H01\tbinaire = source";
        let x = analyser_ligne(l).unwrap();
        assert_eq!(
            (x.nom.as_str(), x.machine.as_str(), x.statut),
            ("G-H01", "acemagic", Statut::Pass)
        );
        assert_eq!(x.detail, "binaire = source");
        assert!(analyser_ligne("PASS G-H01").is_none());
        assert!(analyser_ligne("GATE\tPEUT-ETRE\tacemagic\tX\t").is_none());
        assert!(analyser_ligne("GATE\tPASS\t\tX\t").is_none());
        // détail absent toléré
        assert_eq!(analyser_ligne("GATE\tSKIP\tgamer\tLOI-0").unwrap().detail, "");
    }

    #[test]
    fn jeton_entier_et_alias() {
        let a = Attendue {
            jeton: "EQC-1",
            machines: AM,
            rapide: false,
            tache: "T5",
        };
        assert!(a.couverte_par("EQC-1 (CPU)"));
        assert!(a.couverte_par("eqc-1"));
        assert!(!a.couverte_par("EQC-10"));
        assert!(!a.couverte_par("EQC-1b"));
        let b = Attendue {
            jeton: "BORD-C2a|BORD-C1/C2a",
            machines: AM,
            rapide: false,
            tache: "T3",
        };
        assert!(b.couverte_par("BORD-C1/C2a"));
        let t = Attendue {
            jeton: "LOI-τ|LOI-tau",
            machines: AM,
            rapide: false,
            tache: "T4",
        };
        assert!(t.couverte_par("LOI-tau") && t.couverte_par("LOI-τ"));
        let h = Attendue {
            jeton: "G-H01",
            machines: AM,
            rapide: true,
            tache: "T9",
        };
        assert!(h.couverte_par("G-H01 binaire = source") && !h.couverte_par("XG-H01"));
    }

    #[test]
    fn catalogue_sans_doublon_et_complet_pour_les_13_gates() {
        let mut vus = std::collections::HashSet::new();
        for a in CATALOGUE {
            assert!(vus.insert(a.jeton), "doublon {}", a.jeton);
            assert!(!a.machines.is_empty());
        }
        for n in 1..=13 {
            let nom = format!("G-H{n:02}");
            assert!(CATALOGUE.iter().any(|a| a.jeton == nom), "{nom} absente du catalogue");
        }
        // aucune gate n'est couverte par deux jetons distincts du catalogue
        for a in CATALOGUE {
            let nom = a.jeton.split('|').next().unwrap();
            let n = CATALOGUE.iter().filter(|b| b.couverte_par(nom)).count();
            assert_eq!(n, 1, "{nom} couverte par {n} jetons");
        }
    }

    #[test]
    fn gate_manquante_devient_fail() {
        let machines = vec!["acemagic".to_string()];
        let att = attendues(Mode::Complet, &machines, Some("G-H0"));
        let emises = vec![g("G-H01", "acemagic", Statut::Pass), g("G-H02", "gamer", Statut::Pass)];
        let tout = completer(emises, &att);
        // G-H01 présente ; G-H02 émise seulement par le gamer => manquante sur acemagic.
        assert!(tout
            .iter()
            .any(|x| x.nom == "G-H02" && x.machine == "acemagic" && x.statut == Statut::Fail));
        assert!(!tout.iter().any(|x| x.nom == "G-H01" && x.statut == Statut::Fail));
        assert!(tout.iter().any(|x| x.nom == "G-H09" && x.detail.contains("non émise")));
    }

    #[test]
    fn mode_rapide_ne_reclame_que_les_gates_cpu() {
        let m = vec!["acemagic".to_string(), "gamer".to_string()];
        let r = attendues(Mode::Rapide, &m, None);
        assert!(r.iter().all(|(a, _)| a.rapide));
        assert!(r.iter().any(|(a, _)| a.jeton == "G-H01"));
        assert!(!r.iter().any(|(a, _)| a.jeton == "G-H06"));
        let c = attendues(Mode::Complet, &m, None);
        assert!(c.len() > r.len());
    }

    #[test]
    fn skip_nest_pas_vert() {
        let v = vec![g("A", "acemagic", Statut::Pass), g("B", "gamer", Statut::Skip)];
        assert_eq!(code_de(&v), crate::commun::code::VALIDATION);
        let w = vec![g("A", "acemagic", Statut::Pass), g("B", "gamer", Statut::Pass)];
        assert_eq!(code_de(&w), 0);
        let f = vec![g("A", "acemagic", Statut::Fail)];
        assert_eq!(code_de(&f), crate::commun::code::VALIDATION);
        assert_eq!(
            code_de(&[]),
            crate::commun::code::VALIDATION,
            "aucune gate n'est pas un succès"
        );
    }

    #[test]
    fn json_au_format_gele() {
        let v = vec![
            g("A", "acemagic", Statut::Pass),
            g("B", "gamer", Statut::Skip),
            g("C", "gamer", Statut::Fail),
        ];
        let j: serde_json::Value = serde_json::from_str(&json(&v)).unwrap();
        assert_eq!(j["fail"], 1);
        assert_eq!(j["skip"], 1);
        assert_eq!(j["gates"].as_array().unwrap().len(), 3);
        assert_eq!(j["gates"][1]["statut"], "SKIP");
        for cle in ["nom", "machine", "statut", "detail"] {
            assert!(j["gates"][0].get(cle).is_some(), "clé {cle} absente");
        }
    }

    #[test]
    fn statuts_arbitres_d3_ne_bloquent_pas_et_ne_verdissent_pas() {
        let l = "GATE\tINFO\tacemagic\tOBS-4-L8\t15/20";
        assert_eq!(analyser_ligne(l).unwrap().statut, Statut::Info);
        let l = "GATE\tFAIL-DOCUMENTE\tacemagic\tG-H11\tz = -3,28";
        assert_eq!(analyser_ligne(l).unwrap().statut, Statut::FailDocumente);
        let v = vec![
            g("A", "acemagic", Statut::Pass),
            g("OBS-4-L8", "acemagic", Statut::Info),
            g("G-H11", "acemagic", Statut::FailDocumente),
        ];
        assert_eq!(totaux(&v), (1, 0, 0), "INFO et FAIL-DOCUMENTE ne sont ni PASS, ni FAIL, ni SKIP");
        assert_eq!(totaux_arbitres(&v), (1, 1));
        assert_eq!(code_de(&v), 0, "arbitrés par D3 : non bloquants");
        assert!(tableau(&v).contains("1 INFO, 1 FAIL-DOCUMENTE"));
        let j: serde_json::Value = serde_json::from_str(&json(&v)).unwrap();
        assert_eq!((j["info"].as_u64(), j["fail_documente"].as_u64()), (Some(1), Some(1)));
        assert_eq!(j["gates"][2]["statut"], "FAIL-DOCUMENTE");
        // OBS-4 n'est plus un jeton : OBS-4-L8 et OBS-4-L16 sont deux jetons distincts
        assert!(!CATALOGUE.iter().any(|a| a.jeton == "OBS-4"));
        for nom in ["OBS-4-L8", "OBS-4-L16", "OBS1-MUTANT"] {
            assert_eq!(CATALOGUE.iter().filter(|a| a.couverte_par(nom)).count(), 1, "{nom}");
        }
    }

    #[test]
    fn tableau_contient_les_totaux() {
        let t = tableau(&[g("A", "acemagic", Statut::Pass)]);
        assert!(t.contains("1 PASS, 0 FAIL, 0 SKIP"), "{t}");
    }

    #[test]
    fn analyse_de_sortie_ignore_le_bruit() {
        let s = "bruit\nGATE\tPASS\tacemagic\tX\tok\ncompilation...\nGATE\tFAIL\tgamer\tY\tcassé\n";
        let v = analyser_sortie(s);
        assert_eq!(v.len(), 2);
        assert_eq!(totaux(&v), (1, 1, 0));
    }
}
