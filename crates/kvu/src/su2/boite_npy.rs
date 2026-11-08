// SPDX-License-Identifier: Apache-2.0
//! Bords de boîte lus dans un fichier NPY (`kvu su2 boite --bord-npy`). Propriétaire : WP21. Spécification : `docs/BORD_NPY.md`.
//!
//! Le fichier est un champ de liens du réseau de calcul de la boîte (côté ℓ + 2), au format de `--dump-final` (q64) ou en c16.
//! Il est lu en FLUX : tous ses octets sont hachés (sha256 du fichier), seuls les liens FIXÉS de la boîte sont décodés et validés
//! (règle SU(2) unique de `stockage::npy::decoder_lien`), puis copiés au bit dans le champ de départ par `boite::champ_initial`.
//! Les autres liens (libres, inexistants, de remplissage) ne sont jamais décodés : leur contenu n'a aucun effet.
//! Nom du bord : `npy:<8 premiers hexadécimaux du sha256>` ; `npy` seul désigne l'unique fichier d'un lot.

use super::boite::GeometrieBoite;
use super::boite_run::ChaineBoite;
use crate::commun::{Erreur, Resultat};
use crate::heritage::manifest::{hex, Sha256};
use crate::stockage::npy::{self, FormatChamp};
use serde_json::{json, Value};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Nom (nu) des bords lus ; un bord lu s'appelle `npy:<h8>`.
pub const PREFIXE_NPY: &str = "npy";
/// Liens lus par bloc (q64 : 1 Mio, c16 : 2 Mio par bloc).
const LIENS_PAR_BLOC: usize = 1 << 15;

/// Un bord lu dans un fichier NPY.
#[derive(Clone, Debug, PartialEq)]
pub struct BordNpy {
    /// `npy:<8 premiers hexadécimaux minuscules du sha256>`.
    pub nom: String,
    /// Chemin absolu canonique du fichier lu.
    pub chemin: PathBuf,
    /// sha256 de TOUS les octets du fichier (comme `sha256sum`).
    pub sha256: String,
    /// Taille du fichier (octets).
    pub octets: u64,
    pub format: FormatChamp,
    /// Côté du réseau de calcul (ℓ + 2).
    pub l_calcul: usize,
    /// Nombre de liens fixés de la boîte.
    pub n_liens_fixes: usize,
    /// 4 réels par lien fixé, dans l'ordre croissant de l'indice de lien `li = μ·V + site` (celui de `champ_initial`).
    pub liens: Arc<Vec<f64>>,
}

/// `npy:` suivi des 8 premiers caractères d'un sha256 hexadécimal.
pub fn nom_npy(sha256: &str) -> String {
    format!("{PREFIXE_NPY}:{}", sha256.get(..8).unwrap_or(sha256))
}

/// Vrai si `nom` est exactement `npy:` suivi de 8 hexadécimaux minuscules.
pub fn nom_npy_valide(nom: &str) -> bool {
    nom.strip_prefix("npy:")
        .is_some_and(|h| h.len() == 8 && h.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
}

/// Lecteur qui hache tout ce qui passe.
struct LecteurHache<R: Read> {
    r: R,
    h: Sha256,
}

impl<R: Read> Read for LecteurHache<R> {
    fn read(&mut self, t: &mut [u8]) -> std::io::Result<usize> {
        let n = self.r.read(t)?;
        self.h.update(&t[..n]);
        Ok(n)
    }
}

/// Préfixe le chemin du fichier aux erreurs de format (les autres erreurs portent déjà leur contexte).
fn prefixer(e: Erreur, chemin: &Path) -> Erreur {
    match e {
        Erreur::Format(m) => Erreur::Format(format!("--bord-npy {} : {m}", chemin.display())),
        autre => autre,
    }
}

fn erreur_format(chemin: &Path, m: impl std::fmt::Display) -> Erreur {
    Erreur::Format(format!("--bord-npy {} : {m}", chemin.display()))
}

/// Lit le bord d'une boîte de géométrie `g` dans `chemin` (voir le module). Codes : 4 pour un fichier absent, illisible ou mal
/// formé (en-tête, dtype, forme, taille, lien fixé hors SU(2), NaN, ∞), 2 pour un côté ≠ ℓ + 2.
pub fn lire_bord_npy(chemin: &Path, g: &GeometrieBoite) -> Resultat<BordNpy> {
    let f = File::open(chemin).map_err(|e| Erreur::io(format!("ouverture du bord --bord-npy {}", chemin.display()), e))?;
    let octets = f
        .metadata()
        .map_err(|e| Erreur::io(format!("stat du bord --bord-npy {}", chemin.display()), e))?
        .len();
    let mut r = LecteurHache { r: BufReader::with_capacity(1 << 20, f), h: Sha256::new() };
    let e = npy::lire_entete(&mut r).map_err(|er| prefixer(er, chemin))?;
    let (format, l) = npy::disposition_champ(&e).map_err(|er| prefixer(er, chemin))?;
    if l != g.lc {
        return Err(Erreur::Config(format!(
            "--bord-npy {} : champ de côté {l} ; une boîte de côté ℓ = {} vit dans un réseau de calcul de côté {}",
            chemin.display(),
            g.ell,
            g.lc
        )));
    }
    let v = g.v();
    let total = 4 * v;
    let taille = format.octets_par_lien();
    let attendu = e.octets_entete as u64 + (total as u64) * (taille as u64);
    if octets != attendu {
        return Err(erreur_format(chemin, format!("taille {octets} octets au lieu de {attendu} annoncés par l'en-tête")));
    }
    let mut liens = Vec::with_capacity(4 * g.n_fixes());
    let mut tampon = vec![0u8; LIENS_PAR_BLOC.min(total) * taille];
    let mut fait = 0usize;
    while fait < total {
        let n = LIENS_PAR_BLOC.min(total - fait);
        npy::lire_exact(&mut r, &mut tampon[..n * taille], &format!("lien {fait} sur {total}")).map_err(|er| prefixer(er, chemin))?;
        for (i, lien) in tampon[..n * taille].chunks_exact(taille).enumerate() {
            let li = fait + i;
            let (mu, x) = (li / v, g.coords(li % v));
            if !g.lien_fixe(mu, x) {
                continue; // jamais décodé : seul le sha256 du fichier en dépend
            }
            match npy::decoder_lien(format, lien) {
                Ok(q) => liens.extend_from_slice(&q),
                Err(defaut) => {
                    return Err(erreur_format(
                        chemin,
                        format!("lien fixé {li} (μ = {mu}, x = {x:?}) hors SU(2) : défaut {defaut:e} (tolérance 1e-12 ; NaN et ∞ refusés)"),
                    ))
                }
            }
        }
        fait += n;
    }
    let mut un = [0u8; 1];
    match r.read(&mut un) {
        Ok(0) => {}
        Ok(_) => return Err(erreur_format(chemin, "octets en trop après les données du champ")),
        Err(er) => return Err(Erreur::io(format!("lecture de la fin du bord --bord-npy {}", chemin.display()), er)),
    }
    if liens.len() != 4 * g.n_fixes() {
        return Err(Erreur::Validation(format!(
            "--bord-npy {} : {} liens fixés lus, {} attendus",
            chemin.display(),
            liens.len() / 4,
            g.n_fixes()
        )));
    }
    let LecteurHache { h, .. } = r;
    let sha256 = hex(&h.finalize());
    let canonique = std::fs::canonicalize(chemin)
        .map_err(|er| Erreur::io(format!("chemin canonique du bord --bord-npy {}", chemin.display()), er))?;
    Ok(BordNpy {
        nom: nom_npy(&sha256),
        chemin: canonique,
        sha256,
        octets,
        format,
        l_calcul: l,
        n_liens_fixes: g.n_fixes(),
        liens: Arc::new(liens),
    })
}

/// Liste des noms des bords lus, pour les messages.
pub fn noms(bords: &[BordNpy]) -> String {
    if bords.is_empty() {
        "aucun".to_string()
    } else {
        bords.iter().map(|b| b.nom.as_str()).collect::<Vec<_>>().join(", ")
    }
}

/// Remplace le nom nu `npy` par le nom de l'UNIQUE fichier du lot ; refus (usage) sans fichier ou avec plusieurs.
pub fn resoudre_noms(chaines: &mut [ChaineBoite], bords: &[BordNpy]) -> Resultat<()> {
    for c in chaines.iter_mut().filter(|c| c.bord == PREFIXE_NPY) {
        match bords {
            [] => return Err(Erreur::Config("bord « npy » : aucun --bord-npy donné".into())),
            [b] => c.bord = b.nom.clone(),
            _ => {
                return Err(Erreur::Config(format!(
                    "bord « npy » ambigu : {} fichiers --bord-npy ({}) ; nommer le bord npy:<h8>",
                    bords.len(),
                    noms(bords)
                )))
            }
        }
    }
    Ok(())
}

fn json_bord(b: &BordNpy) -> Value {
    json!({
        "nom": b.nom, "chemin": b.chemin.to_string_lossy(), "sha256": b.sha256, "octets": b.octets,
        "format": b.format.nom(), "l_calcul": b.l_calcul, "n_liens_fixes": b.n_liens_fixes,
    })
}

/// Entrées des bords lus pour `manifest.json` (`bords_npy`) et `parametres.boite.bords_npy` de `kvu-regen/1`.
pub fn json_bords(bords: &[BordNpy]) -> Value {
    Value::Array(bords.iter().map(json_bord).collect())
}

/// Entrées `entrees` de `kvu-regen/1` : les mêmes, avec `"role": "bord-npy"` (contrôlées par `kvu regen` avant tout rejeu).
pub fn json_entrees(bords: &[BordNpy]) -> Value {
    Value::Array(
        bords
            .iter()
            .map(|b| {
                let mut v = json_bord(b);
                v["role"] = json!("bord-npy");
                v
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noms_npy() {
        let sha = "1a2b3c4d5e6f00112233445566778899aabbccddeeff00112233445566778899";
        assert_eq!(nom_npy(sha), "npy:1a2b3c4d");
        assert!(nom_npy_valide("npy:1a2b3c4d"));
        for faux in ["npy", "npy:", "npy:1a2b3c4", "npy:1a2b3c4d5", "npy:1A2B3C4D", "npy:zzzzzzzz", "cold", "npy:1a2b3c4d:7", "NPY:1a2b3c4d"] {
            assert!(!nom_npy_valide(faux), "{faux}");
        }
    }

    #[test]
    fn resolution_du_nom_nu() {
        let g = GeometrieBoite::nouvelle(2).unwrap();
        let b = |h: &str| BordNpy {
            nom: format!("npy:{h}"),
            chemin: PathBuf::from("/x"),
            sha256: format!("{h}{}", "0".repeat(56)),
            octets: 0,
            format: FormatChamp::Q64,
            l_calcul: g.lc,
            n_liens_fixes: g.n_fixes(),
            liens: Arc::new(Vec::new()),
        };
        let mut c = vec![ChaineBoite { bord: "npy".into(), graine: 1 }, ChaineBoite { bord: "cold".into(), graine: 2 }];
        assert!(resoudre_noms(&mut c.clone(), &[]).is_err());
        assert!(resoudre_noms(&mut c.clone(), &[b("00000001"), b("00000002")]).is_err());
        resoudre_noms(&mut c, &[b("0000abcd")]).unwrap();
        assert_eq!(c[0].bord, "npy:0000abcd");
        assert_eq!(c[1].bord, "cold");
        let e = json_entrees(&[b("0000abcd")]);
        assert_eq!(e[0]["role"], "bord-npy");
        assert_eq!(e[0]["format"], "q64");
        assert_eq!(json_bords(&[]), json!([]));
    }
}
