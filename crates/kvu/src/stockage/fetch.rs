//! fetch : rapatrie un ensemble ou un fichier de l'autre machine par le câble 2,5 Gbit/s
//! (rsync -a --partial sur ssh vers l'hôte de câble du profil), vérifie les sha256 du manifeste distant,
//! applique la garde SSD sur la taille distante AVANT de copier. Propriétaire : T8.
//!
//! Déroulement de [`recuperer`] :
//! 1. `ssh <câble> du -sb` : taille distante (et nature : répertoire ou fichier) ;
//! 2. garde SSD locale sur CETTE taille : refus (rc 7) avant le moindre octet copié ;
//! 3. `rsync -a --partial -e 'ssh -o BatchMode=yes'` ;
//! 4. contrôle des sha256 du `manifest.json` copié (répertoire) ou de `sha256sum` distant (fichier seul).
//!    Un écart est une ERREUR, le fichier fautif est SUPPRIMÉ localement et, pour un ensemble, tous ses
//!    `COMPLETE` sont retirés (`FETCH_REFUSE.txt` écrit) : jamais d'ensemble partiel marqué complet.

use super::garde::GardeSsd;
use super::store;
use crate::commun::{Erreur, Resultat};
use crate::heritage::manifest::sha256_fichier;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RapportFetch {
    pub source: String,
    pub destination: PathBuf,
    pub octets: u64,
    pub secondes: f64,
    pub sha_verifies: u64,
}

/// Hôte du câble direct par défaut : KEVINOTRON_CABLE, sinon la clé `cable` du profil de la machine, sinon vide
/// (l'appelant refuse alors : aucun câble n'est configuré, par exemple sur une machine louée).
pub fn hote_cable_defaut() -> String {
    let profil = crate::machine::detection::detecter_rapide();
    hote_cable_depuis(
        std::env::var("KEVINOTRON_CABLE").ok().as_deref(),
        crate::machine::politique(&profil).pair_cable,
    )
}

/// Cœur pur de [`hote_cable_defaut`] : variable `KEVINOTRON_CABLE` > câble de la politique (`kvu.toml`) > vide.
pub fn hote_cable_depuis(env_cable: Option<&str>, pair_cable: Option<String>) -> String {
    if let Some(h) = env_cable.map(str::trim).filter(|h| !h.is_empty()) {
        return h.to_string();
    }
    pair_cable.unwrap_or_default()
}

/// Source distante analysée : `alias:chemin`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceDistante {
    /// Machine désignée (nom d'un profil reconnaissable de `kvu.toml`).
    pub alias: String,
    /// Chemin pour le shell distant : absolu, ou relatif au répertoire personnel.
    pub chemin: String,
    /// La source se terminait par « / » : on veut le CONTENU du répertoire.
    pub contenu: bool,
}

/// Racine du store de la machine distante : clé `store` de son profil ; `~/x` devient `x`, relatif au répertoire
/// personnel distant (le shell distant est lancé dans ce répertoire).
fn racine_store_distant(alias: &str) -> String {
    let store = alias
        .parse::<crate::commun::Machine>()
        .map(|m| crate::config::actuelle().profil(m).store)
        .unwrap_or_default();
    match store.strip_prefix("~/") {
        Some(relatif) => relatif.to_string(),
        None => store,
    }
}

/// Alias de machine admis : profils reconnaissables de `kvu.toml`.
fn alias_connu(alias: &str) -> bool {
    alias
        .parse::<crate::commun::Machine>()
        .is_ok_and(|m| crate::config::actuelle().machines_reconnaissables().contains(&m))
}

fn noms_alias() -> String {
    crate::config::actuelle()
        .machines_reconnaissables()
        .iter()
        .map(|m| m.nom())
        .collect::<Vec<_>>()
        .join("|")
}

fn caractere_permis(c: char) -> bool {
    c.is_ascii_alphanumeric() || "_./+=,@%~-".contains(c)
}

/// Analyse `<machine>:<chemin absolu ou relatif à ~>` ou `<machine>:store/<id>`.
/// Les chemins sont restreints à un jeu de caractères sûr (aucune citation shell nécessaire) et sans `..`.
pub fn analyser_source(source: &str) -> Resultat<SourceDistante> {
    let Some((alias, reste)) = source.split_once(':') else {
        return Err(Erreur::Config(format!(
            "source « {source} » : forme attendue <machine>:<chemin> ou <machine>:store/<id> (machines : {})",
            noms_alias()
        )));
    };
    if !alias_connu(alias) {
        return Err(Erreur::Config(format!(
            "machine « {alias} » inconnue ({})",
            noms_alias()
        )));
    }
    let contenu = reste.ends_with('/');
    let reste = reste.trim_end_matches('/');
    let reste = reste.strip_prefix("~/").unwrap_or(reste);
    if reste.is_empty() || reste.starts_with('-') || !reste.chars().all(caractere_permis) {
        return Err(Erreur::Config(format!(
            "chemin distant « {reste} » refusé (caractères permis : lettres, chiffres, _ . / + = , @ % ~ -)"
        )));
    }
    if reste.split('/').any(|c| c == "..") {
        return Err(Erreur::Config(format!(
            "chemin distant « {reste} » refusé : « .. » interdit"
        )));
    }
    let chemin = match reste.strip_prefix("store/") {
        Some(id) if !id.is_empty() => format!("{}/{id}", racine_store_distant(alias)),
        _ if reste == "store" => racine_store_distant(alias),
        _ => reste.to_string(),
    };
    Ok(SourceDistante {
        alias: alias.to_string(),
        chemin,
        contenu,
    })
}

/// Commande rsync effectivement lancée (pure, testable).
///
/// `source` : forme acceptée par [`analyser_source`] ; terminée par « / », elle copie le CONTENU du
/// répertoire dans `destination` (créée au besoin). Une source invalide donne une commande vide.
pub fn commande_rsync(source: &str, destination: &Path, hote_cable: &str) -> Vec<String> {
    let Ok(s) = analyser_source(source) else {
        return Vec::new();
    };
    let mut distant = format!("{hote_cable}:{}", s.chemin);
    let mut dest = destination.to_string_lossy().into_owned();
    if s.contenu {
        distant.push('/');
        if !dest.ends_with('/') {
            dest.push('/');
        }
    }
    vec![
        "rsync".into(),
        "-a".into(),
        "--partial".into(),
        "-e".into(),
        "ssh -o BatchMode=yes".into(),
        distant,
        dest,
    ]
}

/// `ssh -o BatchMode=yes -o ConnectTimeout=10 <hôte> <commande>` (pure).
pub fn commande_ssh(hote_cable: &str, commande_distante: &str) -> Vec<String> {
    [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        hote_cable,
        commande_distante,
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Commande distante qui rend la taille (`du -sb`) puis « DIR » ou « FILE » (pure ; `chemin` déjà validé).
pub fn commande_taille(chemin: &str) -> String {
    format!("du -sb -- '{chemin}' && if test -d '{chemin}'; then echo DIR; else echo FILE; fi")
}

/// Analyse la sortie de [`commande_taille`] : (octets, est_un_repertoire).
pub fn analyser_taille(sortie: &str) -> Resultat<(u64, bool)> {
    let mut lignes = sortie.lines();
    let octets = lignes
        .next()
        .and_then(|l| l.split_whitespace().next())
        .and_then(|t| t.parse::<u64>().ok())
        .ok_or_else(|| Erreur::Format(format!("réponse de du -sb illisible : {sortie:?}")))?;
    match lignes.next().map(str::trim) {
        Some("DIR") => Ok((octets, true)),
        Some("FILE") => Ok((octets, false)),
        _ => Err(Erreur::Format(format!(
            "nature distante (DIR|FILE) absente de la réponse : {sortie:?}"
        ))),
    }
}

fn executer(cmd: &[String], quoi: &str) -> Resultat<String> {
    let sortie = Command::new(&cmd[0])
        .args(&cmd[1..])
        .output()
        .map_err(|e| Erreur::io(format!("lancement de {}", cmd[0]), e))?;
    if !sortie.status.success() {
        return Err(Erreur::io(
            quoi.to_string(),
            std::io::Error::other(format!(
                "{} a rendu {} : {}",
                cmd[0],
                sortie.status,
                String::from_utf8_lossy(&sortie.stderr).trim()
            )),
        ));
    }
    Ok(String::from_utf8_lossy(&sortie.stdout).into_owned())
}

/// `source` = « <machine>:<chemin absolu ou relatif à ~> » ou « <machine>:store/<id> ».
///
/// `destination` : un répertoire pour un ensemble (reçoit son contenu) ; pour un fichier seul, un répertoire
/// existant (le fichier y est déposé sous son nom) ou le chemin du fichier à créer.
pub fn recuperer(source: &str, destination: &Path, garde: &GardeSsd, hote_cable: &str) -> Resultat<RapportFetch> {
    let src = analyser_source(source)?;
    let debut = Instant::now();

    let sortie = executer(
        &commande_ssh(hote_cable, &commande_taille(&src.chemin)),
        &format!("sonde de taille distante {source}"),
    )?;
    let (octets, est_repertoire) = analyser_taille(&sortie)?;

    // Garde SSD sur la taille DISTANTE : refus avant tout octet copié.
    garde.verifier(destination, octets)?;

    let source_rsync = format!("{}:{}{}", src.alias, src.chemin, if est_repertoire { "/" } else { "" });
    if est_repertoire {
        std::fs::create_dir_all(destination)
            .map_err(|e| Erreur::io(format!("création de {}", destination.display()), e))?;
    }
    let cmd = commande_rsync(&source_rsync, destination, hote_cable);
    executer(&cmd, &format!("rsync {source}"))?;

    let sha_verifies = if est_repertoire {
        verifier_ensemble(destination, source)?
    } else {
        verifier_fichier(&src, destination, hote_cable)?
    };
    Ok(RapportFetch {
        source: source.to_string(),
        destination: destination.to_path_buf(),
        octets,
        secondes: debut.elapsed().as_secs_f64(),
        sha_verifies,
    })
}

/// Contrôle des sha256 du manifeste copié ; supprime les fichiers fautifs et rend une erreur au moindre écart.
fn verifier_ensemble(destination: &Path, source: &str) -> Resultat<u64> {
    let r = match store::verifier_repertoire(destination) {
        Ok(r) => r,
        Err(Erreur::Format(m)) => {
            eprintln!("fetch : {source} n'a pas de manifest.json, aucun sha256 vérifiable ({m})");
            return Ok(0);
        }
        Err(e) => return Err(e),
    };
    if !r.ok() {
        let mut supprimes = Vec::new();
        for e in &r.ecarts {
            if e.nature == "sha256" && std::fs::remove_file(&e.chemin).is_ok() {
                supprimes.push(e.chemin.display().to_string());
            }
        }
        let premier = &r.ecarts[0];
        let message = format!(
            "fetch {source} : {} écart(s) de sha256 (par exemple {} : {}) ; fichier(s) fautif(s) supprimé(s) : [{}]",
            r.ecarts.len(),
            premier.chemin.display(),
            premier.detail,
            supprimes.join(", ")
        );
        let retires = invalider_ensemble(destination, &message)?;
        return Err(Erreur::Validation(format!(
            "{message} ; ensemble PARTIEL : {} COMPLETE retiré(s), {NOM_FETCH_REFUSE} écrit",
            retires.len()
        )));
    }
    for s in &r.sans_complete {
        eprintln!(
            "fetch : ATTENTION {} n'a pas de COMPLETE (run en cours ou interrompu côté source)",
            s.display()
        );
    }
    Ok(r.fichiers_verifies)
}

/// Marqueur écrit à la racine d'un ensemble dont le fetch a été refusé.
pub const NOM_FETCH_REFUSE: &str = "FETCH_REFUSE.txt";

/// Un fetch refusé laisse un ensemble PARTIEL : aucun `COMPLETE` ne doit y subsister (vérificateur du 04/10 :
/// après un sha256 faux, la destination gardait COMPLETE et les autres fichiers, donc un ensemble partiel marqué
/// complet). Retire tous les `COMPLETE` sous `dir` (trois niveaux : ensemble, chaînes `cNNN/`, sous-runs) et écrit
/// [`NOM_FETCH_REFUSE`] avec la raison. Rend les marqueurs retirés.
pub fn invalider_ensemble(dir: &Path, raison: &str) -> Resultat<Vec<PathBuf>> {
    fn parcourir(d: &Path, profondeur: usize, retires: &mut Vec<PathBuf>) {
        let c = d.join("COMPLETE");
        if c.is_file() && std::fs::remove_file(&c).is_ok() {
            retires.push(c);
        }
        if profondeur == 0 {
            return;
        }
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                if e.file_type().is_ok_and(|t| t.is_dir()) {
                    parcourir(&e.path(), profondeur - 1, retires);
                }
            }
        }
    }
    let mut retires = Vec::new();
    parcourir(dir, 2, &mut retires);
    let texte = format!(
        "fetch REFUSÉ ({}) : ensemble PARTIEL, à ne pas utiliser.\n{raison}\nCOMPLETE retirés : {}\n",
        crate::machine::file::horodatage_iso(crate::machine::file::maintenant_unix()),
        retires.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
    );
    crate::heritage::manifest::ecrire_atomique(&dir.join(NOM_FETCH_REFUSE), texte.as_bytes())
        .map_err(|e| Erreur::io(format!("écriture de {}", dir.join(NOM_FETCH_REFUSE).display()), e))?;
    Ok(retires)
}

/// Fichier seul : sha256 distant (`sha256sum`) contre sha256 local.
fn verifier_fichier(src: &SourceDistante, destination: &Path, hote_cable: &str) -> Resultat<u64> {
    let nom = src.chemin.rsplit('/').next().unwrap_or(&src.chemin);
    let local = if destination.is_dir() {
        destination.join(nom)
    } else {
        destination.to_path_buf()
    };
    let sortie = executer(
        &commande_ssh(hote_cable, &format!("sha256sum -- '{}'", src.chemin)),
        "sha256sum distant",
    )?;
    let distant = sortie.split_whitespace().next().unwrap_or("").to_string();
    let obtenu = sha256_fichier(&local).map_err(|e| Erreur::io(format!("sha256 {}", local.display()), e))?;
    if distant != obtenu {
        let _ = std::fs::remove_file(&local);
        return Err(Erreur::Validation(format!(
            "fetch {}:{} : sha256 distant {distant} différent du local {obtenu} ; fichier fautif {} supprimé",
            src.alias,
            src.chemin,
            local.display()
        )));
    }
    Ok(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyse_des_sources() {
        let s = analyser_source("gamer:/home/utilisateur/ym4_calcul/x").unwrap();
        assert_eq!(
            (s.alias.as_str(), s.chemin.as_str(), s.contenu),
            ("gamer", "/home/utilisateur/ym4_calcul/x", false)
        );
        let s = analyser_source("gamer:~/ym4_calcul/x/").unwrap();
        assert_eq!((s.chemin.as_str(), s.contenu), ("ym4_calcul/x", true));
        assert_eq!(
            analyser_source("gamer:store/run42").unwrap().chemin,
            "kevinotron_store/run42"
        );
        assert_eq!(
            analyser_source("acemagic:store/run42").unwrap().chemin,
            "/chemin/vers/projet/kevinotron_store/run42"
        );
        for mauvais in [
            "gamer",
            "autre:/x",
            "gamer:",
            "gamer:../x",
            "gamer:/a/../b",
            "gamer:-rf",
            "gamer:/a b",
            "gamer:/a;rm",
            "gamer:$(x)",
        ] {
            assert!(analyser_source(mauvais).is_err(), "{mauvais} aurait dû être refusé");
        }
    }

    /// Vérificateur du 04/10 : après un fetch refusé, plus aucun COMPLETE dans la destination.
    #[test]
    fn ensemble_partiel_sans_complete() {
        let d = std::env::temp_dir().join(format!("kvu_fetch_partiel_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("c000")).unwrap();
        for f in ["COMPLETE", "manifest.json", "c000/COMPLETE", "c000/therm.csv"] {
            std::fs::write(d.join(f), "x").unwrap();
        }
        let r = invalider_ensemble(&d, "sha256 faux").unwrap();
        assert_eq!(r.len(), 2);
        assert!(!d.join("COMPLETE").exists() && !d.join("c000/COMPLETE").exists());
        assert!(d.join("c000/therm.csv").exists(), "les autres fichiers restent (diagnostic)");
        let t = std::fs::read_to_string(d.join(NOM_FETCH_REFUSE)).unwrap();
        assert!(t.contains("PARTIEL") && t.contains("sha256 faux"), "{t}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn commande_rsync_golden() {
        assert_eq!(
            commande_rsync("gamer:store/run42/", Path::new("/data/run42"), "gamer-eth"),
            [
                "rsync",
                "-a",
                "--partial",
                "-e",
                "ssh -o BatchMode=yes",
                "gamer-eth:kevinotron_store/run42/",
                "/data/run42/"
            ]
        );
        assert_eq!(
            commande_rsync("gamer:/home/k/x.npy", Path::new("/data"), "192.0.2.2"),
            [
                "rsync",
                "-a",
                "--partial",
                "-e",
                "ssh -o BatchMode=yes",
                "192.0.2.2:/home/k/x.npy",
                "/data"
            ]
        );
        assert!(commande_rsync("gamer:../x", Path::new("/data"), "gamer-eth").is_empty());
    }

    #[test]
    fn commandes_ssh_et_taille() {
        let c = commande_ssh("gamer-eth", &commande_taille("kevinotron_store/r"));
        assert_eq!(&c[..5], ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]);
        assert_eq!(c[5], "gamer-eth");
        assert!(c[6].starts_with("du -sb -- 'kevinotron_store/r' && "));
        assert_eq!(
            analyser_taille("4096\tkevinotron_store/r\nDIR\n").unwrap(),
            (4096, true)
        );
        assert_eq!(analyser_taille("12\tx\nFILE\n").unwrap(), (12, false));
        assert!(analyser_taille("du: impossible\n").is_err());
        assert!(analyser_taille("12\tx\n").is_err());
    }
}
