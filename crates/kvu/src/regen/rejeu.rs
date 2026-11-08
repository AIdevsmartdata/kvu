// SPDX-License-Identifier: Apache-2.0
//! Rejeu d'une chaîne à partir de son manifeste (`kvu regen`) : reconstruit la commande, exige un binaire admis, rejoue
//! SOUS VERROU GPU dans un répertoire de sortie NEUF, compare les sha256 fichier par fichier, écrit `regen_verifiee` dans
//! une COPIE du manifeste (jamais dans l'ensemble d'origine). Rend rc 0 (identique) ou 8 (différence) par l'appelant.

use super::adaptateurs::{adapter, reconnaitre, Hachage, Source};
use super::classer::{gbit_requise, Contexte};
use super::manifeste::{est_sous, maintenant_iso, sha_fichier};
use super::schema::valider_manifeste;
use crate::commun::{Erreur, Resultat};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Options de `kvu regen`.
#[derive(Clone, Debug, Default)]
pub struct OptionsRejeu {
    /// Répertoire des originaux (défaut : le répertoire du manifeste d'origine).
    pub comparer: Option<PathBuf>,
    /// Répertoire de sortie, NEUF.
    pub sortie: PathBuf,
    /// `tests/anchors.sha256` : lignes `<backend> <fichier> <sha256>`.
    pub ancres: Option<PathBuf>,
    /// Fichier de preuves G-BIT (lignes `PASS <ID> binaire_sha256=<sha> …`).
    pub preuves: Option<PathBuf>,
    /// Délai maximal du rejeu (s) ; défaut : fenêtre GPU de la politique (600 s).
    pub delai_s: Option<u64>,
    /// Contrôle négatif REGEN-3 : graine + 1.
    pub mutant_graine: bool,
    /// Binaire exécutant (tests) ; défaut : le binaire courant.
    pub executant: Option<PathBuf>,
}

/// Verdict pour un fichier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResultatFichier {
    pub nom: String,
    pub attendu: Option<String>,
    pub obtenu: Option<String>,
    pub ancre: Option<String>,
    /// `OK`, `DIFFERENT`, `ABSENT`, `ANCRE_DIFFERE`.
    pub etat: &'static str,
}

/// Rapport d'un rejeu.
#[derive(Clone, Debug)]
pub struct RapportRejeu {
    pub manifeste_regen: PathBuf,
    pub rapport_json: PathBuf,
    pub fichiers: Vec<ResultatFichier>,
    pub identiques: usize,
    pub ancres_comparees: usize,
    pub ancres_identiques: usize,
    pub rc_enfant: i32,
    pub duree_s: f64,
    pub binaire_executant: String,
    pub admission: String,
}

impl RapportRejeu {
    pub fn tout_identique(&self) -> bool {
        self.rc_enfant == 0
            && self.identiques == self.fichiers.len()
            && !self.fichiers.is_empty()
            && self.ancres_comparees == self.ancres_identiques
    }
}

/// Lit `anchors.sha256` : (backend, fichier) → sha256, hors lignes `binary` et commentaires.
pub fn lire_ancres(texte: &str) -> BTreeMap<(String, String), String> {
    let mut m = BTreeMap::new();
    for l in texte.lines() {
        let l = l.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let j: Vec<&str> = l.split_whitespace().collect();
        if j.len() == 3 && j[1] != "binary" && super::manifeste::est_sha256(j[2]) {
            m.insert((j[0].to_string(), j[1].to_string()), j[2].to_string());
        }
    }
    m
}

/// Remplace la valeur qui suit `--graine` par graine + 1 (contrôle négatif REGEN-3).
pub fn muter_graine(commande: &[String]) -> Resultat<Vec<String>> {
    let mut c = commande.to_vec();
    let i = c
        .iter()
        .position(|a| a == "--graine")
        .ok_or_else(|| Erreur::Config("commande sans --graine : mutant impossible".into()))?;
    let g: u64 = c
        .get(i + 1)
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| Erreur::Config("--graine sans valeur entière".into()))?;
    c[i + 1] = (g + 1).to_string();
    Ok(c)
}

fn admission(m: &Value, sha_exe: &str, ctx: &Contexte) -> Resultat<String> {
    let attendu = m["binaire_sha256"].as_str().unwrap_or("");
    if attendu == sha_exe {
        return Ok(format!("binaire exécutant identique au binaire attendu ({}…)", &sha_exe[..8]));
    }
    let backend = m["backend"].as_str().unwrap_or("");
    let options = m["parametres"]["options"].as_str().unwrap_or("");
    match gbit_requise(backend, options) {
        Some(g) if ctx.gbit_verte(g) => Ok(format!(
            "binaire exécutant {}… différent du binaire attendu {}… mais {g} PASS pour ce binaire",
            &sha_exe[..8],
            attendu.get(..8).unwrap_or(attendu)
        )),
        Some(g) => Err(Erreur::Config(format!(
            "binaire non admis : {}… ≠ binaire attendu {}… et aucune preuve PASS {g} pour ce binaire (--preuve-gbit)",
            &sha_exe[..8],
            attendu.get(..8).unwrap_or(attendu)
        ))),
        None => Err(Erreur::Config(format!(
            "binaire non admis : {}… ≠ binaire attendu et aucune G-BIT connue pour ({backend}, {options})",
            &sha_exe[..8]
        ))),
    }
}

/// Entrées du manifeste (`entrees` portant un `chemin`, ex. bords `--bord-npy`) : chaque fichier doit exister et avoir le sha256
/// consigné, sinon rien n'est rejoué (absent : rc 4 ; modifié : rc 8). Les entrées sans `chemin` ne sont pas contrôlées ici.
pub fn verifier_entrees(m: &Value) -> Resultat<()> {
    for e in m["entrees"].as_array().into_iter().flatten() {
        let Some(chemin) = e["chemin"].as_str() else { continue };
        let attendu = e["sha256"].as_str().unwrap_or("");
        let obtenu = sha_fichier(Path::new(chemin))?;
        if obtenu != attendu {
            return Err(Erreur::Validation(format!(
                "entrée modifiée : {chemin} : sha256 {obtenu} ≠ {attendu} consigné au manifeste ; rien n'est rejoué"
            )));
        }
    }
    Ok(())
}

/// Rejoue la chaîne décrite par `manifeste` (voir le module).
pub fn rejouer(manifeste: &Path, opts: &OptionsRejeu) -> Resultat<RapportRejeu> {
    let exe: PathBuf = match &opts.executant {
        Some(p) => p.clone(),
        None => std::env::current_exe().map_err(|e| Erreur::io("chemin du binaire courant", e))?,
    };
    let sha_exe = sha_fichier(&exe)?;
    let mut ctx = Contexte::nouveau();
    if let Some(p) = &opts.preuves {
        ctx.charger_preuves(p, &sha_exe)?;
    }
    // Manifeste -> kvu-regen/1 (les originaux sont hachés sur disque plus bas, seulement pour les fichiers listés).
    let octets_m = std::fs::read(manifeste).map_err(|e| Erreur::io(format!("lecture de {}", manifeste.display()), e))?;
    let brut: Value = serde_json::from_slice(&octets_m).map_err(|e| Erreur::Config(format!("{} : JSON invalide : {e}", manifeste.display())))?;
    let natif = reconnaitre(&brut) == Some(Source::Regen);
    let mut m = adapter(manifeste, &mut ctx, Hachage::Omettre)?;
    let erreurs = valider_manifeste(&m);
    if !erreurs.is_empty() {
        return Err(Erreur::Config(format!(
            "manifeste invalide au schéma kvu-regen/1 ({} erreur(s)) : {}",
            erreurs.len(),
            erreurs.iter().take(5).cloned().collect::<Vec<_>>().join(" ; ")
        )));
    }
    if m["statut"] != "COMPLETE" {
        return Err(Erreur::Config(format!("chaîne non COMPLETE ({}) : rien à rejouer", m["statut"])));
    }
    verifier_entrees(&m)?;
    let adm = admission(&m, &sha_exe, &ctx)?;

    // Répertoire des originaux et sortie : jamais dans un ensemble existant.
    let rep_manifeste = manifeste.parent().unwrap_or_else(|| Path::new("."));
    let comparer: Option<PathBuf> = opts.comparer.clone().or_else(|| if natif { None } else { Some(rep_manifeste.to_path_buf()) });
    if (!natif && est_sous(&opts.sortie, rep_manifeste)) || comparer.as_deref().is_some_and(|c| est_sous(&opts.sortie, c)) {
        return Err(Erreur::Config(format!(
            "REFUS : la sortie {} se trouve sous le répertoire de l'ensemble d'origine (écriture interdite)",
            opts.sortie.display()
        )));
    }
    if opts.sortie.exists() {
        return Err(Erreur::Config(format!(
            "REFUS D'ÉCRASER : {} existe déjà (la sortie d'un rejeu doit être neuve)",
            opts.sortie.display()
        )));
    }

    // Sha256 attendus : manifeste, sinon originaux sur disque (lecture seule).
    let fichiers = m["fichiers"].as_array().cloned().unwrap_or_default();
    if fichiers.is_empty() {
        return Err(Erreur::Config("manifeste sans fichiers à comparer".into()));
    }
    let mut attendus: Vec<(String, String, u64)> = Vec::new();
    for f in &fichiers {
        let nom = f["nom"].as_str().unwrap_or("").to_string();
        let sha = match f["sha256"].as_str() {
            Some(s) => s.to_string(),
            None => {
                let c = comparer.as_ref().ok_or_else(|| {
                    Erreur::Config(format!("{nom} : sha256 absent du manifeste et aucun répertoire à comparer (--comparer)"))
                })?;
                sha_fichier(&c.join(&nom))?
            }
        };
        attendus.push((nom, sha, f["octets"].as_u64().unwrap_or(0)));
    }
    let ancres = match &opts.ancres {
        Some(p) => lire_ancres(&std::fs::read_to_string(p).map_err(|e| Erreur::io(format!("lecture de {}", p.display()), e))?),
        None => BTreeMap::new(),
    };

    // Commande reconstruite.
    let mut commande: Vec<String> = m["commande"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
        .unwrap_or_default();
    if opts.mutant_graine {
        commande = muter_graine(&commande)?;
    }
    let total: u64 = attendus.iter().map(|a| a.2).sum();
    let parent_sortie = opts.sortie.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent_sortie).map_err(|e| Erreur::io(format!("création de {}", parent_sortie.display()), e))?;
    crate::stockage::GardeSsd::machine().verifier(parent_sortie, total + (1 << 20))?;
    std::fs::create_dir(&opts.sortie).map_err(|e| Erreur::io(format!("création de la sortie {}", opts.sortie.display()), e))?;
    let rejeu_dir = opts.sortie.join("rejeu");
    let log = opts.sortie.join("rejeu.log");

    // Verrou GPU de la machine, tenu pendant tout le rejeu (le fils le voit comme tenu par un ancêtre).
    let profil = crate::machine::detecter();
    let pol = crate::machine::politique(&profil);
    let _verrou = if m["backend"] != "cpu" {
        Some(crate::machine::VerrouGpu::nouveau(&pol.verrou_gpu).acquerir(Duration::from_secs(pol.attente_verrou_s))?)
    } else {
        None
    };

    let t0 = Instant::now();
    let sortie_log = std::fs::File::create(&log).map_err(|e| Erreur::io(format!("création de {}", log.display()), e))?;
    let sortie_err = sortie_log.try_clone().map_err(|e| Erreur::io("duplication du journal", e))?;
    let mut enfant = std::process::Command::new(&exe)
        .args(&commande)
        .arg("--outdir")
        .arg(&rejeu_dir)
        .stdin(std::process::Stdio::null())
        .stdout(sortie_log)
        .stderr(sortie_err)
        .spawn()
        .map_err(|e| Erreur::io(format!("lancement de {}", exe.display()), e))?;
    let delai = Duration::from_secs(opts.delai_s.unwrap_or(pol.fenetre_banc_s));
    let rc_enfant = loop {
        match enfant.try_wait().map_err(|e| Erreur::io("attente du rejeu", e))? {
            Some(st) => break st.code().unwrap_or(-1),
            None if t0.elapsed() > delai => {
                let _ = enfant.kill();
                let _ = enfant.wait();
                return Err(Erreur::Ressources(format!(
                    "rejeu interrompu après {} s (fenêtre maximale) ; journal : {}",
                    delai.as_secs(),
                    log.display()
                )));
            }
            None => std::thread::sleep(Duration::from_millis(200)),
        }
    };
    let duree_s = t0.elapsed().as_secs_f64();

    // Comparaison fichier par fichier.
    let backend = m["backend"].as_str().unwrap_or("").to_string();
    let mut res = Vec::new();
    let (mut identiques, mut anc_c, mut anc_i) = (0usize, 0usize, 0usize);
    for (nom, attendu, _) in &attendus {
        let chemin = rejeu_dir.join(nom);
        let obtenu = if chemin.is_file() { Some(sha_fichier(&chemin)?) } else { None };
        let ancre = ancres.get(&(backend.clone(), nom.clone())).cloned();
        let mut etat = match &obtenu {
            None => "ABSENT",
            Some(o) if o == attendu => "OK",
            Some(_) => "DIFFERENT",
        };
        if let (Some(a), Some(o)) = (&ancre, &obtenu) {
            anc_c += 1;
            if a == o {
                anc_i += 1;
            } else if etat == "OK" {
                etat = "ANCRE_DIFFERE";
            }
        } else if ancre.is_some() {
            anc_c += 1;
        }
        if etat == "OK" {
            identiques += 1;
        }
        res.push(ResultatFichier { nom: nom.clone(), attendu: Some(attendu.clone()), obtenu, ancre, etat });
    }

    // Copie du manifeste avec regen_verifiee (jamais dans l'ensemble d'origine).
    let tout_ok = rc_enfant == 0 && identiques == res.len() && anc_c == anc_i;
    if let Some(arr) = m["fichiers"].as_array_mut() {
        for (f, (_, sha, _)) in arr.iter_mut().zip(&attendus) {
            f["sha256"] = Value::String(sha.clone());
        }
    }
    if m["fichiers_hachage"] == "non_calcule" {
        m["fichiers_hachage"] = Value::String("disque".into());
    }
    m["regen_verifiee"] = json!({
        "date": maintenant_iso(),
        "machine": profil.machine.nom(),
        "resultat": if tout_ok { "identique" } else { "different" },
    });
    let manifeste_regen = opts.sortie.join("manifest_regen.json");
    let texte = serde_json::to_string_pretty(&m).map_err(|e| Erreur::Format(e.to_string()))?;
    std::fs::write(&manifeste_regen, texte + "\n").map_err(|e| Erreur::io(format!("écriture de {}", manifeste_regen.display()), e))?;

    let rapport = RapportRejeu {
        manifeste_regen,
        rapport_json: opts.sortie.join("rapport_regen.json"),
        fichiers: res,
        identiques,
        ancres_comparees: anc_c,
        ancres_identiques: anc_i,
        rc_enfant,
        duree_s,
        binaire_executant: sha_exe,
        admission: adm,
    };
    let rj = json!({
        "manifeste": manifeste.to_string_lossy(),
        "sortie": opts.sortie.to_string_lossy(),
        "commande": commande,
        "mutant_graine": opts.mutant_graine,
        "binaire_executant": rapport.binaire_executant,
        "admission": rapport.admission,
        "rc_enfant": rapport.rc_enfant,
        "duree_s": rapport.duree_s,
        "fichiers_total": rapport.fichiers.len(),
        "fichiers_identiques": rapport.identiques,
        "ancres_comparees": rapport.ancres_comparees,
        "ancres_identiques": rapport.ancres_identiques,
        "resultat": if tout_ok { "identique" } else { "different" },
        "fichiers": rapport.fichiers.iter().map(|f| json!({
            "nom": f.nom, "etat": f.etat, "attendu": f.attendu, "obtenu": f.obtenu, "ancre": f.ancre
        })).collect::<Vec<_>>(),
    });
    std::fs::write(&rapport.rapport_json, serde_json::to_string_pretty(&rj).unwrap_or_default() + "\n")
        .map_err(|e| Erreur::io(format!("écriture de {}", rapport.rapport_json.display()), e))?;
    // Les fichiers rejoués restent dans le scratch (gros) : leur suppression est l'affaire du lanceur.
    Ok(rapport)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graine_mutee_de_un() {
        let c: Vec<String> = "su2 gen --L 12 --graine 26095612011 --start hot".split_whitespace().map(String::from).collect();
        let m = muter_graine(&c).unwrap();
        assert_eq!(m[5], "26095612012");
        assert_eq!(&c[..5], &m[..5]);
        assert_eq!(c[6..], m[6..]);
        assert!(muter_graine(&["su2".to_string()]).is_err());
    }

    #[test]
    fn ancres_lues_sans_les_lignes_binary() {
        let t = "# commentaire\nhip binary d62ff7d2a9cfc861e8c02faf4b1e284aaed7da8a2b9cd42f7aec14d335af93cf\nhip cfg_002000.npy 357246ceab4588bb2fce78c2b769271e775dd8419adbcb390fce45d59aa77c7e\n";
        let a = lire_ancres(t);
        assert_eq!(a.len(), 1);
        assert!(a.contains_key(&("hip".to_string(), "cfg_002000.npy".to_string())));
    }

    #[test]
    fn admission_binaire() {
        let sha = "1".repeat(64);
        let m = json!({"binaire_sha256": "2".repeat(64), "backend": "hip", "parametres": {"options": "ref-k1"}});
        let sans = Contexte::nouveau();
        assert!(admission(&m, &sha, &sans).is_err(), "binaire différent sans preuve : refusé");
        let avec = Contexte::nouveau().avec_gbit("G-BIT-K1");
        assert!(admission(&m, &sha, &avec).unwrap().contains("G-BIT-K1 PASS"));
        let autre_gbit = Contexte::nouveau().avec_gbit("G-BIT-HIP-PROD");
        assert!(admission(&m, &sha, &autre_gbit).is_err(), "une autre G-BIT ne suffit pas");
        let identique = json!({"binaire_sha256": sha, "backend": "hip", "parametres": {"options": "ref-k1"}});
        assert!(admission(&identique, &sha, &sans).unwrap().contains("identique"));
    }

    #[test]
    fn refuse_une_sortie_sous_l_ensemble_d_origine() {
        let d = std::env::temp_dir().join(format!("kvu_rejeu_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let man = d.join("manifest.json");
        std::fs::write(
            &man,
            br#"{"schema":1,"engine":"su2_hip","source_hash":"ed1e0fce255cb15d640ab603b3af0b6c50075bbb65252ce5d7e351abd3061e30","backend":"HIP gfx1151 FP64","rng":"x","l":4,"beta":2.5,"seed":5,"start":"Hot","therm_steps":10,"measurements":10,"skip_steps":1,"or_per_step":4,"dump_every":5,"statut":"COMPLETE"}"#,
        )
        .unwrap();
        std::fs::write(d.join("therm.csv"), b"t\n").unwrap();
        let sha = sha_fichier(&std::env::current_exe().unwrap()).unwrap();
        let preuves = d.join("preuves.txt");
        std::fs::write(&preuves, format!("PASS G-BIT-HIP-PROD binaire_sha256={sha}\n")).unwrap();
        let opts = OptionsRejeu { sortie: d.join("sortie_interdite"), preuves: Some(preuves), ..Default::default() };
        let e = rejouer(&man, &opts).unwrap_err().to_string();
        assert!(e.contains("REFUS") && e.contains("sous le répertoire"), "{e}");
        assert!(!d.join("sortie_interdite").exists());
        let _ = std::fs::remove_dir_all(&d);
    }
}
