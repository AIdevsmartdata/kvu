//! Manifeste de provenance unifié, au-dessus de heritage::manifest::RepertoireRun (contrat K8 :
//! répertoire NEUF, manifest.json à l'ouverture, COMPLETE en dernier si rc = 0). Propriétaire : T8.
//!
//! Schéma « kevinotron-unifie/1 ». Champs écrits à l'ouverture :
//! - hérités de K8 : `moteur`, `version`, `binaire`, `argv`, `cwd`, `hote`, `pid`, `debut_unix`, `statut`, `rc`, `fin_unix` ;
//! - `schema` (= kevinotron-unifie/1) et `schema_k8` (= kevinotron-provenance/1) ;
//! - `source_hash` : recette `kvu-source/1` (reproductible : `validation/historique/source_hash_kvu.py`), avec
//!   `source_hash_recette` et `source_hash_nfichiers` ;
//! - `binaire_sha256` (/proc/self/exe) et `git_head` (KEV_GIT_HEAD embarqué à la compilation) ;
//! - `machine`, `backend`, `arch`, `versions` (hip_runtime, hiprtc, cuda_pilote, nvrtc…), `politique` ;
//! - `noyaux_embarques` : [{nom, sha256}] des 11 sources GPU du crate ;
//! - `modules_compiles` : [ProvenanceCompil] (source assemblée, parties, options, arguments, sha256 du code objet) ;
//! - `verrou` (pris_ici | ancetre | partage | ignore | non_determine), `garde_ssd` {seuil_go, libre_go_debut, chemin} ;
//! - `fichiers` : [{nom, sha256, octets}] des fichiers produits.
//!
//! Le reste (précision, mode, graines, options numériques…) s'ajoute par [`Manifeste::champ`].
//! Les valeurs structurées sont du JSON brut inséré tel quel dans le manifeste K8 (via `Valeur::Nombre`,
//! qui imprime son contenu sans le modifier) : aucun changement au contrat K8.

use super::garde::GardeSsd;
use crate::commun::{Erreur, Resultat};
use crate::gpu::ProvenanceCompil;
use crate::heritage::manifest::{self, champs_de_base, sha256_fichier, ErreurRepertoire, RepertoireRun, Valeur};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Schéma du manifeste unifié.
pub const SCHEMA_UNIFIE: &str = "kevinotron-unifie/1";
/// Octets réservés à l'ouverture pour le manifeste lui-même (vérifiés par la garde SSD).
pub const OCTETS_MANIFESTE: u64 = 1 << 20;
/// Recette de source `kvu-source/1` (DECISIONS D2, WP1) : nom, listes de fichiers et description, définis dans
/// [`super::recette`] (module std seulement, partagé avec `build.rs`) et réexportés ici.
pub use super::recette::{DOSSIERS_KVU_SOURCE_1, FICHIERS_KVU_SOURCE_1, NOM_RECETTE};
/// Description de la recette, écrite dans le manifeste (champ `source_hash_recette`).
pub const RECETTE_UNIFIEE_TEXTE: &str = super::recette::TEXTE_RECETTE;

/// Clés que `champ` ne peut pas réécrire (invariants du contrat K8 et empreintes de provenance).
const CLES_RESERVEES: &[&str] = &[
    "schema",
    "schema_k8",
    "statut",
    "rc",
    "fin_unix",
    "binaire_sha256",
    "source_hash",
    "noyaux_embarques",
    "modules_compiles",
    "fichiers",
];

/// Empreinte `kvu-source/1` embarquée dans CE binaire par crates/kvu/build.rs (nom hérité : « unifiee »).
pub fn source_hash_unifiee() -> Option<&'static str> {
    option_env!("KEVINOTRON_SOURCE_HASH_UNIFIE")
}

/// Nombre de fichiers couverts par l'empreinte embarquée.
pub fn source_nfichiers_unifiee() -> Option<&'static str> {
    option_env!("KEVINOTRON_SOURCE_NFICHIERS_UNIFIE")
}

/// git HEAD embarqué à la compilation (KEV_GIT_HEAD), « non_enregistre » sinon.
pub fn git_head() -> &'static str {
    option_env!("KEV_GIT_HEAD").unwrap_or("non_enregistre")
}

/// sha256 du binaire en cours d'exécution (/proc/self/exe).
pub fn sha256_binaire() -> Resultat<String> {
    sha256_fichier(Path::new("/proc/self/exe"))
        .or_else(|_| std::env::current_exe().and_then(|p| sha256_fichier(&p)))
        .map_err(|e| Erreur::io("sha256 du binaire (/proc/self/exe)", e))
}

/// Valeur JSON brute pour un champ du manifeste K8.
fn brut(v: &Value) -> Valeur {
    Valeur::Nombre(v.to_string())
}

fn erreur_repertoire(e: ErreurRepertoire) -> Erreur {
    match e {
        ErreurRepertoire::Vide => Erreur::Config("chemin de répertoire de run vide".into()),
        ErreurRepertoire::Existe(p) => Erreur::Config(format!(
            "REFUS D'ÉCRASER : {} existe déjà (un répertoire de run doit être neuf ; choisir un autre chemin)",
            p.display()
        )),
        ErreurRepertoire::Io(p, e) => Erreur::io(format!("répertoire de run {}", p.display()), e),
    }
}

/// Champs du manifeste unifié (hors garde SSD, ajoutée par l'appelant).
fn champs_unifies(moteur: &str, argv: &[String], garde: &GardeSsd, dir: &Path) -> manifest::Champs {
    let mut c = champs_de_base(moteur, env!("CARGO_PKG_VERSION"), argv);
    c.set("schema", Valeur::texte(SCHEMA_UNIFIE));
    c.set("schema_k8", Valeur::texte(manifest::SCHEMA));
    c.set("source_hash", Valeur::option_texte(source_hash_unifiee()));
    c.set("source_hash_recette", Valeur::texte(RECETTE_UNIFIEE_TEXTE));
    c.set(
        "source_hash_nfichiers",
        Valeur::option_texte(source_nfichiers_unifiee()),
    );
    // champs_de_base a recalculé le hachage historique de la recette Complete : sans objet ici.
    c.set("source_hash_build_sh", Valeur::Nul);
    c.set("binaire_sha256", Valeur::option_texte(sha256_binaire().ok().as_deref()));
    c.set("git_head", Valeur::texte(git_head()));

    let profil = crate::machine::detecter();
    c.set("machine", Valeur::texte(profil.machine.nom()));
    match crate::machine::choisir_backend(&profil) {
        Ok(b) => {
            c.set("backend", Valeur::texte(b.nom()));
            let arch = profil.gpus.iter().find(|g| g.backend == b).map(|g| g.arch.as_str());
            c.set("arch", Valeur::option_texte(arch));
        }
        Err(e) => {
            c.set("backend", Valeur::Nul);
            c.set("arch", Valeur::Nul);
            c.set("backend_erreur", Valeur::texte(e.to_string()));
        }
    }
    let versions: serde_json::Map<String, Value> = profil
        .versions
        .iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect();
    c.set("versions", brut(&Value::Object(versions)));
    c.set(
        "politique",
        brut(&serde_json::to_value(crate::machine::politique(&profil)).unwrap_or(Value::Null)),
    );

    let noyaux: Vec<Value> = crate::noyaux::empreintes()
        .into_iter()
        .map(|(nom, sha)| json!({"nom": nom, "sha256": sha}))
        .collect();
    c.set("noyaux_embarques", brut(&Value::Array(noyaux)));
    c.set("modules_compiles", brut(&Value::Array(Vec::new())));

    let verrou = if std::env::var("KEVINOTRON_VERROU").is_ok_and(|v| v == "ignorer") {
        "ignore"
    } else {
        "non_determine"
    };
    c.set("verrou", Valeur::texte(verrou));
    let libre = GardeSsd::libre_go(dir).ok();
    c.set(
        "garde_ssd",
        brut(&json!({"seuil_go": garde.seuil_go, "libre_go_debut": libre, "chemin": dir.display().to_string()})),
    );
    c.set("fichiers", brut(&Value::Array(Vec::new())));
    c
}

/// Suffixe du manifeste d'un FICHIER de sortie isolé (`cond sweep --out Y.npy` ⇒ `Y.npy.manifest.json`).
pub const SUFFIXE_MANIFESTE_FICHIER: &str = ".manifest.json";

/// Manifeste d'un fichier de sortie isolé (une sortie qui n'est pas un répertoire de run) : mêmes champs de
/// provenance que [`Manifeste`] (binaire, source_hash, git_head, machine, noyaux embarqués, politique, garde SSD),
/// plus `fichiers` = [{nom, sha256, octets}] de `sortie` et les `ajouts` (backend effectif, modules compilés,
/// paramètres). Écrit atomiquement à côté de la sortie, APRÈS elle ; rend son chemin.
pub fn ecrire_manifeste_fichier(
    sortie: &Path,
    moteur: &str,
    argv: &[String],
    ajouts: &[(&str, Value)],
) -> Resultat<PathBuf> {
    let dir = sortie.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let mut c = champs_unifies(moteur, argv, &GardeSsd::machine(), dir);
    let octets = std::fs::metadata(sortie).map_err(|e| Erreur::io(format!("stat {}", sortie.display()), e))?.len();
    let sha = sha256_fichier(sortie).map_err(|e| Erreur::io(format!("sha256 {}", sortie.display()), e))?;
    let nom = sortie.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    c.set("fichiers", brut(&json!([{"nom": nom, "sha256": sha, "octets": octets}])));
    c.set("statut", Valeur::texte("termine"));
    c.set("rc", Valeur::entier(0));
    c.set("fin_unix", Valeur::flottant(manifest::maintenant_unix()));
    for (k, v) in ajouts {
        if matches!(*k, "schema" | "schema_k8" | "statut" | "rc" | "fichiers" | "binaire_sha256" | "source_hash") {
            return Err(Erreur::Config(format!("la clé « {k} » du manifeste est réservée")));
        }
        c.set(k, brut(v));
    }
    let mut chemin = sortie.as_os_str().to_owned();
    chemin.push(SUFFIXE_MANIFESTE_FICHIER);
    let chemin = PathBuf::from(chemin);
    manifest::ecrire_atomique(&chemin, c.json().as_bytes())
        .map_err(|e| Erreur::io(format!("écriture de {}", chemin.display()), e))?;
    Ok(chemin)
}

/// Manifeste d'un run : un répertoire NEUF, `en_cours` à l'ouverture, COMPLETE en dernier si rc = 0.
pub struct Manifeste {
    pub rep: RepertoireRun,
}

impl Manifeste {
    /// Ouvre un répertoire de run NEUF (refus s'il existe) et écrit le manifeste « en_cours ».
    /// La garde SSD de la machine est consultée AVANT toute création (rc 7 si le disque est sous le seuil).
    pub fn ouvrir(dir: &Path, moteur: &str, argv: &[String]) -> Resultat<Manifeste> {
        Self::ouvrir_avec_garde(dir, moteur, argv, &GardeSsd::machine(), 0)
    }

    /// Comme [`Manifeste::ouvrir`], avec une garde explicite et la PROJECTION totale du run (octets,
    /// voir `garde::projeter`) : si `libre − projection < seuil`, rien n'est créé et l'erreur SsdPlein
    /// (rc 7) est rendue. Le seuil et le libre de départ sont consignés dans `garde_ssd`.
    pub fn ouvrir_avec_garde(
        dir: &Path,
        moteur: &str,
        argv: &[String],
        garde: &GardeSsd,
        projection_octets: u64,
    ) -> Resultat<Manifeste> {
        garde.verifier(dir, projection_octets.saturating_add(OCTETS_MANIFESTE))?;
        let champs = champs_unifies(moteur, argv, garde, dir);
        let rep = RepertoireRun::ouvrir(dir, champs).map_err(erreur_repertoire)?;
        Ok(Manifeste { rep })
    }

    fn ecrire(&self) -> Resultat<()> {
        self.rep
            .ecrire_manifeste()
            .map_err(|e| Erreur::io(format!("écriture de {}", self.rep.manifeste().display()), e))
    }

    /// Contenu JSON brut d'un champ liste, ou un tableau vide.
    fn liste(&self, cle: &str) -> Vec<Value> {
        match self.rep.champs.get(cle) {
            Some(Valeur::Nombre(s)) => serde_json::from_str::<Vec<Value>>(s).unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    fn fixer_liste(&mut self, cle: &str, v: Vec<Value>) {
        self.rep.champs.set(cle, brut(&Value::Array(v)));
    }

    /// Ajoute les provenances de modules compilés (une entrée par source chargée). Un module identique
    /// (même nom, même source, même code objet) n'est consigné qu'une fois. Écrit le manifeste.
    pub fn ajouter_compil(&mut self, p: &[ProvenanceCompil]) -> Resultat<()> {
        let mut v = self.liste("modules_compiles");
        for m in p {
            let e = serde_json::to_value(m).map_err(|e| Erreur::Format(format!("provenance de compilation : {e}")))?;
            if !v.contains(&e) {
                v.push(e);
            }
        }
        self.fixer_liste("modules_compiles", v);
        self.ecrire()
    }

    /// Champ libre : si `json` est un JSON valide (nombre, booléen, objet, liste, chaîne entre guillemets),
    /// il est inséré tel quel (normalisé sur une ligne) ; sinon c'est un texte. Écrit le manifeste.
    /// Les clés d'invariants (`statut`, `rc`, `source_hash`, `fichiers`…) ne peuvent pas être réécrites.
    pub fn champ(&mut self, cle: &str, json: &str) -> Resultat<()> {
        if CLES_RESERVEES.contains(&cle) {
            return Err(Erreur::Config(format!("la clé « {cle} » du manifeste est réservée")));
        }
        let valeur = match serde_json::from_str::<Value>(json) {
            Ok(Value::String(s)) => Valeur::Texte(s),
            Ok(Value::Null) => Valeur::Nul,
            Ok(v) => brut(&v),
            Err(_) => Valeur::texte(json),
        };
        self.rep.champs.set(cle, valeur);
        self.ecrire()
    }

    /// Consigne l'état du verrou GPU : `pris_ici`, `ancetre`, `partage` (LOCK_SH sous gpu_partage), `ignore` ou `non_determine`.
    pub fn verrou(&mut self, etat: &str) -> Resultat<()> {
        match etat {
            "pris_ici" | "ancetre" | "partage" | "ignore" | "non_determine" => {
                self.rep.champs.set("verrou", Valeur::texte(etat));
                self.ecrire()
            }
            autre => Err(Erreur::Config(format!(
                "état de verrou inconnu « {autre} » (pris_ici|ancetre|partage|ignore|non_determine)"
            ))),
        }
    }

    /// Nom sous lequel un fichier est consigné : relatif au répertoire du run s'il est dedans, sinon complet.
    fn nom_consigne(&self, chemin: &Path) -> String {
        match chemin.strip_prefix(&self.rep.dir) {
            Ok(r) => r.to_string_lossy().into_owned(),
            Err(_) => chemin.to_string_lossy().into_owned(),
        }
    }

    /// Enregistre un fichier produit (sha256 calculé ici, par lecture du fichier). Le manifeste est réécrit à
    /// `terminer` ou à `ecrire_manifeste` (pas à chaque fichier : un run de 10⁴ dumps ne le réécrit pas 10⁴ fois).
    pub fn fichier(&mut self, chemin: &Path) -> Resultat<()> {
        let meta = std::fs::metadata(chemin).map_err(|e| Erreur::io(format!("stat {}", chemin.display()), e))?;
        let sha = sha256_fichier(chemin).map_err(|e| Erreur::io(format!("sha256 {}", chemin.display()), e))?;
        self.fichier_connu(chemin, &sha, meta.len())
    }

    /// Enregistre un fichier dont le sha256 est déjà connu (calculé pendant l'écriture :
    /// `npy::ecrire_champ_hache`). Évite de relire plusieurs Go.
    pub fn fichier_connu(&mut self, chemin: &Path, sha256: &str, octets: u64) -> Resultat<()> {
        if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(Erreur::Config(format!("sha256 invalide pour {}", chemin.display())));
        }
        let mut v = self.liste("fichiers");
        let nom = self.nom_consigne(chemin);
        v.retain(|e| e.get("nom").and_then(Value::as_str) != Some(nom.as_str()));
        v.push(json!({"nom": nom, "sha256": sha256.to_ascii_lowercase(), "octets": octets}));
        self.fixer_liste("fichiers", v);
        Ok(())
    }

    /// Réécrit manifest.json maintenant (point de contrôle, sans terminer le run).
    pub fn ecrire_manifeste(&self) -> Resultat<()> {
        self.ecrire()
    }

    /// Fichiers consignés : (nom, sha256, octets).
    pub fn fichiers(&self) -> Vec<(String, String, u64)> {
        self.liste("fichiers")
            .iter()
            .filter_map(|e| {
                Some((
                    e.get("nom")?.as_str()?.to_string(),
                    e.get("sha256")?.as_str()?.to_string(),
                    e.get("octets")?.as_u64()?,
                ))
            })
            .collect()
    }

    /// Termine : statut, rc ; COMPLETE seulement si rc = 0.
    ///
    /// Avant COMPLETE, chaque fichier consigné doit exister avec la taille consignée : sinon le run est
    /// clos en échec (jamais de COMPLETE) et l'erreur `Validation` est rendue.
    pub fn terminer(mut self, rc: i32) -> Resultat<()> {
        if rc == 0 {
            let mut ecarts = Vec::new();
            for (nom, _, octets) in self.fichiers() {
                let chemin = if Path::new(&nom).is_absolute() {
                    PathBuf::from(&nom)
                } else {
                    self.rep.dir.join(&nom)
                };
                match std::fs::metadata(&chemin) {
                    Ok(m) if m.len() == octets => {}
                    Ok(m) => ecarts.push(format!("{nom} : {} octets au lieu de {octets}", m.len())),
                    Err(_) => ecarts.push(format!("{nom} : absent")),
                }
            }
            if !ecarts.is_empty() {
                let raison = format!("fichiers consignés altérés : {}", ecarts.join(" ; "));
                self.rep
                    .terminer_echec(rc, &raison)
                    .map_err(|e| Erreur::io("clôture du manifeste", e))?;
                return Err(Erreur::Validation(raison));
            }
        }
        self.rep
            .terminer(rc)
            .map_err(|e| Erreur::io(format!("clôture de {}", self.rep.manifeste().display()), e))
    }

    /// Clôture en échec quel que soit rc (jamais de COMPLETE), avec la raison consignée.
    pub fn echouer(mut self, rc: i32, raison: &str) -> Resultat<()> {
        self.rep
            .terminer_echec(rc, raison)
            .map_err(|e| Erreur::io("clôture du manifeste", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empreinte_embarquee_presente() {
        let h = source_hash_unifiee().expect("injectée par crates/kvu/build.rs");
        assert_eq!(h.len(), 64);
        // recette kvu-source/1 : crates/kvu/{src,noyaux}/** + 5 fichiers (94 à l'import de WP1) ; l'ancienne recette
        // Unifiee couvrait en plus src/** du moteur historique (plus de 100 fichiers), absent de ce dépôt.
        assert!(source_nfichiers_unifiee().unwrap().parse::<usize>().unwrap() > 50);
    }

    #[test]
    fn cles_reservees_refusees() {
        let scratch = super::super::scratch_tests("prov_unit");
        let base = scratch.chemin().to_path_buf();
        let mut m = Manifeste::ouvrir_avec_garde(&base.join("run"), "test", &["x".into()], &GardeSsd::nouvelle(0.0), 0)
            .unwrap();
        assert!(m.champ("statut", "termine").is_err());
        assert!(m.champ("fichiers", "[]").is_err());
        m.champ("precision", "f64").unwrap();
        m.champ("graines", "[1, 2]").unwrap();
        assert!(m.verrou("n_importe_quoi").is_err());
        let texte = std::fs::read_to_string(m.rep.manifeste()).unwrap();
        let v: Value = serde_json::from_str(&texte).unwrap();
        assert_eq!(v["precision"], "f64");
        assert_eq!(v["graines"], json!([1, 2]));
        assert_eq!(v["schema"], SCHEMA_UNIFIE);
        m.terminer(0).unwrap();
    }
}
