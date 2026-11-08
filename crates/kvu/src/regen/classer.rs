// SPDX-License-Identifier: Apache-2.0
//! Classes de régénération A / B / C (ARCHITECTURE §7.2, règle gelée de GATES.md §1/WP4 et du brief WP4).
//!
//! - **A** : manifeste complet + binaire ancré (le fichier du binaire cité est retrouvé avec le même sha256) ou G-BIT
//!   correspondante PASS **pour le binaire qui classe**, produit sur la machine principale
//!   (profil NOM_PROFIL_ACEMAGIC) ;
//! - **B** : produit hors machine principale (machine secondaire, machine louée) ;
//! - **C** : binaire ou graines inconnus, adaptateur non reconnu, chaîne incomplète, ou référence de la publication.
//!
//! La classification est PRUDENTE : dans le doute, C. Un manifeste sans `binaire_sha256` ou sans adaptateur reconnu n'est
//! jamais A (REGEN-4) ; un manifeste `su2_hip/1` n'est A que par l'ancre (`binaire_sha256` fourni par l'adaptateur
//! = `d62ff7d2…`) ET si G-BIT-HIP-PROD est PASS pour le binaire courant.

use crate::commun::{Erreur, Resultat, NOM_PROFIL_ACEMAGIC, NOM_PROFIL_INCONNUE};
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};

pub const ID_GBIT_HIP_PROD: &str = "G-BIT-HIP-PROD";
pub const ID_GBIT_CUDA: &str = "G-BIT-CUDA";
pub const ID_GBIT_K1: &str = "G-BIT-K1";

/// Classe de régénération.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Classe {
    A,
    B,
    C,
}

impl Classe {
    pub fn nom(self) -> &'static str {
        match self {
            Classe::A => "A",
            Classe::B => "B",
            Classe::C => "C",
        }
    }
}

impl fmt::Display for Classe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.nom())
    }
}

/// Résultat d'un classement : classe et raisons lisibles.
#[derive(Clone, Debug, PartialEq)]
pub struct Classement {
    pub classe: Classe,
    pub raisons: Vec<String>,
}

/// Début (8 caractères au plus) d'une empreinte, pour les messages.
fn court(s: &str) -> &str {
    s.get(..8).unwrap_or(s)
}

fn c(raison: impl Into<String>) -> Classement {
    Classement { classe: Classe::C, raisons: vec![raison.into()] }
}

/// Chaîne de référence de la publication (A) : ses sorties sont dans `tests/anchors.sha256` ; classe C, à garder.
pub struct AnchePublication {
    pub id_gbit: &'static str,
    pub backend: &'static str,
    pub l: u64,
    pub beta: f64,
    pub graine: u64,
    pub therm: u64,
    pub nmeas: u64,
}

/// Les deux chaînes ancrées de la publication (GATES G-BIT-HIP-PROD et G-BIT-CUDA).
pub const ANCRES_PUBLICATION: &[AnchePublication] = &[
    AnchePublication { id_gbit: ID_GBIT_HIP_PROD, backend: "hip", l: 12, beta: 2.5, graine: 26_095_612_011, therm: 5000, nmeas: 2000 },
    AnchePublication { id_gbit: ID_GBIT_CUDA, backend: "cuda", l: 8, beta: 2.8, graine: 27_092_708_011, therm: 2000, nmeas: 20000 },
];

/// Ancre de publication à laquelle correspond ce manifeste (paramètres et graine identiques), s'il y en a une.
pub fn ancre_publication(m: &Value) -> Option<&'static AnchePublication> {
    let p = m.pointer("/parametres")?;
    let l = p.get("L")?.as_u64()?;
    let beta = p.get("beta")?.as_f64()?;
    let therm = p.get("therm")?.as_u64()?;
    let nmeas = p.get("nmeas")?.as_u64()?;
    let graine = m.pointer("/graines/cle")?.as_u64()?;
    let backend = m.get("backend")?.as_str()?;
    ANCRES_PUBLICATION.iter().find(|a| {
        a.backend == backend && a.l == l && (a.beta - beta).abs() < 1e-12 && a.graine == graine && a.therm == therm && a.nmeas == nmeas
    })
}

/// G-BIT qui valide le binaire exécutant pour ce couple (backend, options numériques).
pub fn gbit_requise(backend: &str, options: &str) -> Option<&'static str> {
    match (backend, options) {
        ("hip", "compat-prod") => Some(ID_GBIT_HIP_PROD),
        ("cuda", "compat-prod") => Some(ID_GBIT_CUDA),
        ("hip", "ref-k1") => Some(ID_GBIT_K1),
        _ => None,
    }
}

/// Contexte de classement : preuves de G-BIT (valables pour UN binaire) et cache des sha256 de binaires.
#[derive(Default)]
pub struct Contexte {
    gbit_vertes: BTreeSet<String>,
    cache: HashMap<PathBuf, Option<String>>,
}

impl Contexte {
    pub fn nouveau() -> Contexte {
        Contexte::default()
    }

    /// Déclare une G-BIT verte (tests ; en production, passer par [`Contexte::charger_preuves`]).
    pub fn avec_gbit(mut self, id: &str) -> Contexte {
        self.gbit_vertes.insert(id.to_string());
        self
    }

    pub fn gbit_verte(&self, id: &str) -> bool {
        self.gbit_vertes.contains(id)
    }

    pub fn gbit_vertes(&self) -> Vec<String> {
        self.gbit_vertes.iter().cloned().collect()
    }

    /// Lit un fichier de preuves : lignes `PASS <ID> … binaire_sha256=<64 hex> …`. Seules comptent les lignes dont le sha256
    /// est celui du binaire qui classe/rejoue (`sha_binaire`) : une preuve d'un autre binaire est ignorée. Rend le nombre
    /// d'identifiants retenus.
    pub fn charger_preuves(&mut self, fichier: &Path, sha_binaire: &str) -> Resultat<usize> {
        let texte = std::fs::read_to_string(fichier)
            .map_err(|e| Erreur::io(format!("lecture des preuves G-BIT {}", fichier.display()), e))?;
        let mut n = 0;
        for id in ids_prouves(&texte, sha_binaire) {
            if self.gbit_vertes.insert(id) {
                n += 1;
            }
        }
        Ok(n)
    }

    /// sha256 du fichier (mémorisé) ; `None` s'il n'existe pas ou est illisible.
    pub fn sha_fichier(&mut self, p: &Path) -> Option<String> {
        if let Some(v) = self.cache.get(p) {
            return v.clone();
        }
        let v = if p.is_file() { super::manifeste::sha_fichier(p).ok() } else { None };
        self.cache.insert(p.to_path_buf(), v.clone());
        v
    }
}

/// Identifiants G-BIT prouvés par `texte` pour le binaire `sha_binaire` (pure, testée).
pub fn ids_prouves(texte: &str, sha_binaire: &str) -> Vec<String> {
    let mut ids = Vec::new();
    for ligne in texte.lines() {
        let mut jetons = ligne.split_whitespace();
        if jetons.next() != Some("PASS") {
            continue;
        }
        let Some(id) = jetons.next() else { continue };
        let attendu = format!("binaire_sha256={sha_binaire}");
        if jetons.any(|j| j == attendu) {
            ids.push(id.to_string());
        }
    }
    ids
}

/// Classe un manifeste `kvu-regen/1` (adapté ou natif). Les décisions n'utilisent que des champs du manifeste, le contexte
/// et, pour les manifestes `su2_gen/1`, le fichier du binaire cité (`commande_origine[0]`) lu en lecture seule.
pub fn classer(m: &Value, ctx: &mut Contexte) -> Classement {
    let txt = |p: &str| m.pointer(p).and_then(Value::as_str);
    if !matches!(txt("/origine/schema"), Some("su2_hip/1") | Some("su2_gen/1") | Some("kvu-regen/1")) {
        return c("adaptateur non reconnu (REGEN-4) : jamais A");
    }
    if txt("/statut") != Some("COMPLETE") {
        return c(format!("chaîne non COMPLETE ({})", txt("/statut").unwrap_or("statut absent")));
    }
    let Some(binaire) = txt("/binaire_sha256") else {
        return c("binaire_sha256 inconnu (REGEN-4) : jamais A");
    };
    if m.pointer("/graines/cle").and_then(Value::as_u64).is_none() {
        return c("graine inconnue : non régénérable");
    }
    if let Some(a) = ancre_publication(m) {
        return c(format!(
            "référence de la publication : chaîne ancrée {} (tests/anchors.sha256), à garder",
            a.id_gbit
        ));
    }
    match txt("/machine/nom") {
        Some(n) if n == NOM_PROFIL_ACEMAGIC => {}
        Some(n) if n == NOM_PROFIL_INCONNUE => return c("machine de production inconnue"),
        None => return c("machine de production inconnue"),
        Some(autre) => {
            return Classement {
                classe: Classe::B,
                raisons: vec![format!(
                    "produit hors {} ({autre}) : régénération statistique, pas de bit-identité inter-machines",
                    NOM_PROFIL_ACEMAGIC
                )],
            }
        }
    }
    let backend = txt("/backend").unwrap_or("");
    let options = txt("/parametres/options").unwrap_or("");
    let gbit = gbit_requise(backend, options);
    let gbit_ok = gbit.is_some_and(|g| ctx.gbit_verte(g));
    match txt("/binaire_provenance") {
        Some("ancre_publication") => {
            if gbit_ok {
                Classement {
                    classe: Classe::A,
                    raisons: vec![format!(
                        "binaire d'ancre {} fourni par l'adaptateur ; {} PASS pour le binaire courant ; produit sur {}",
                        court(binaire),
                        gbit.unwrap_or(""),
                        NOM_PROFIL_ACEMAGIC
                    )],
                }
            } else {
                c(format!(
                    "binaire d'ancre non confirmé : {} sans preuve PASS pour le binaire courant",
                    gbit.unwrap_or("G-BIT inconnue")
                ))
            }
        }
        Some("manifeste") => {
            let chemin = m.pointer("/commande_origine/0").and_then(Value::as_str);
            let retrouve = chemin.is_some_and(|ch| ctx.sha_fichier(Path::new(ch)).as_deref() == Some(binaire));
            if retrouve {
                Classement {
                    classe: Classe::A,
                    raisons: vec![format!(
                        "binaire ancré : {} retrouvé avec le même sha256 ({}…) ; manifeste complet ; produit sur {}",
                        chemin.unwrap_or(""),
                        court(binaire),
                        NOM_PROFIL_ACEMAGIC
                    )],
                }
            } else if gbit_ok {
                Classement {
                    classe: Classe::A,
                    raisons: vec![format!(
                        concat!(
                            "binaire du manifeste ({}…) absent du disque, mais {} PASS pour le binaire courant ; ",
                            "produit sur {}"
                        ),
                        court(binaire),
                        gbit.unwrap_or(""),
                        NOM_PROFIL_ACEMAGIC
                    )],
                }
            } else {
                c(format!(
                    "binaire {}… introuvable à l'emplacement cité et {} sans preuve PASS : non ancré",
                    court(binaire),
                    gbit.unwrap_or("G-BIT inconnue")
                ))
            }
        }
        _ => c("provenance du binaire non établie"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn base(source: &str, prov: &str, machine: &str, options: &str) -> Value {
        json!({
            "origine": {"schema": source, "manifeste": "m", "sha256_manifeste": "a".repeat(64)},
            "statut": "COMPLETE",
            "binaire_sha256": "d62ff7d2".to_string() + &"0".repeat(56),
            "binaire_provenance": prov,
            "backend": "hip",
            "machine": {"nom": machine},
            "parametres": {"L": 16, "beta": 2.3, "therm": 100, "nmeas": 10, "options": options},
            "graines": {"cle": 5u64},
            "commande_origine": ["/n/existe/pas"]
        })
    }

    #[test]
    fn regen4_sans_adaptateur_ou_sans_binaire_jamais_a() {
        let mut ctx = Contexte::nouveau().avec_gbit(ID_GBIT_HIP_PROD).avec_gbit(ID_GBIT_K1);
        let inconnu = json!({"schema": 1, "engine": "autre", "statut": "COMPLETE"});
        assert_eq!(classer(&inconnu, &mut ctx).classe, Classe::C);
        let mut m = base("su2_hip/1", "ancre_publication", "acemagic", "compat-prod");
        m["binaire_sha256"] = Value::Null;
        let r = classer(&m, &mut ctx);
        assert_eq!(r.classe, Classe::C, "{r:?}");
        let mut m = base("su2_hip/1", "ancre_publication", "acemagic", "compat-prod");
        m.as_object_mut().unwrap().remove("binaire_sha256");
        assert_eq!(classer(&m, &mut ctx).classe, Classe::C);
    }

    #[test]
    fn regen4_su2_hip_a_seulement_par_l_ancre() {
        let m = base("su2_hip/1", "ancre_publication", "acemagic", "compat-prod");
        let mut sans = Contexte::nouveau();
        assert_eq!(classer(&m, &mut sans).classe, Classe::C, "sans preuve G-BIT-HIP-PROD");
        let mut avec = Contexte::nouveau().avec_gbit(ID_GBIT_HIP_PROD);
        assert_eq!(classer(&m, &mut avec).classe, Classe::A);
        // une autre G-BIT verte ne suffit pas
        let mut autre = Contexte::nouveau().avec_gbit(ID_GBIT_K1);
        assert_eq!(classer(&m, &mut autre).classe, Classe::C);
    }

    #[test]
    fn chaine_ancree_de_la_publication_est_c_meme_avec_preuve() {
        let mut m = base("su2_hip/1", "ancre_publication", "acemagic", "compat-prod");
        m["parametres"] = json!({"L": 12, "beta": 2.5, "therm": 5000, "nmeas": 2000, "options": "compat-prod"});
        m["graines"] = json!({"cle": 26_095_612_011u64});
        let mut ctx = Contexte::nouveau().avec_gbit(ID_GBIT_HIP_PROD);
        let r = classer(&m, &mut ctx);
        assert_eq!(r.classe, Classe::C, "{r:?}");
        assert!(r.raisons[0].contains("publication"));
    }

    #[test]
    fn hors_acemagic_est_b_et_incomplet_est_c() {
        let mut ctx = Contexte::nouveau();
        let m = base("su2_gen/1", "manifeste", "gamer", "ref-k1");
        assert_eq!(classer(&m, &mut ctx).classe, Classe::B);
        let mut m = base("su2_gen/1", "manifeste", "acemagic", "ref-k1");
        m["statut"] = json!("INCOMPLETE");
        assert_eq!(classer(&m, &mut ctx).classe, Classe::C);
        let mut m = base("su2_gen/1", "manifeste", "acemagic", "ref-k1");
        m["graines"] = json!({"cle": null});
        assert_eq!(classer(&m, &mut ctx).classe, Classe::C);
    }

    #[test]
    fn su2_gen_a_si_binaire_retrouve_ou_gbit_k1() {
        let dir = std::env::temp_dir().join(format!("kvu_classer_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = dir.join("faux_su2_gen");
        std::fs::write(&bin, b"binaire jouet").unwrap();
        let sha = super::super::manifeste::sha_octets(b"binaire jouet");
        let mut m = base("su2_gen/1", "manifeste", "acemagic", "ref-k1");
        m["binaire_sha256"] = json!(sha);
        m["commande_origine"] = json!([bin.to_string_lossy()]);
        let mut ctx = Contexte::nouveau();
        assert_eq!(classer(&m, &mut ctx).classe, Classe::A);
        // binaire modifié : sha différent, non retrouvé, pas de G-BIT -> C
        m["binaire_sha256"] = json!("b".repeat(64));
        let mut ctx = Contexte::nouveau();
        assert_eq!(classer(&m, &mut ctx).classe, Classe::C);
        // introuvable mais G-BIT-K1 verte -> A
        m["commande_origine"] = json!(["/n/existe/pas"]);
        let mut ctx = Contexte::nouveau().avec_gbit(ID_GBIT_K1);
        assert_eq!(classer(&m, &mut ctx).classe, Classe::A);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn preuves_liees_au_binaire() {
        let sha = "c".repeat(64);
        let autre = "d".repeat(64);
        let t = format!(
            "# commentaire\nPASS G-BIT-HIP-PROD binaire_sha256={sha} 3/3\nPASS G-BIT-K1 binaire_sha256={autre}\nFAIL G-BIT-CUDA binaire_sha256={sha}\nPASS G-BIT-CUDA sans_sha\n"
        );
        assert_eq!(ids_prouves(&t, &sha), vec!["G-BIT-HIP-PROD".to_string()]);
        assert_eq!(ids_prouves(&t, &autre), vec!["G-BIT-K1".to_string()]);
        assert!(ids_prouves(&t, &"e".repeat(64)).is_empty());
    }

    #[test]
    fn gbit_requise_table() {
        assert_eq!(gbit_requise("hip", "compat-prod"), Some(ID_GBIT_HIP_PROD));
        assert_eq!(gbit_requise("hip", "ref-k1"), Some(ID_GBIT_K1));
        assert_eq!(gbit_requise("cuda", "compat-prod"), Some(ID_GBIT_CUDA));
        assert_eq!(gbit_requise("cpu", "ref-k1"), None);
    }
}
