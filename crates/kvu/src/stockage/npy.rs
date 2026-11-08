//! NPY v1.0 : champs SU(2) (c16 compat su2_hip, q64 quaternions) et tableaux f64/f32/u32.
//! Écriture atomique (.part, fsync, rename, fsync du répertoire), garde SSD avant écriture.
//! Propriétaire : T8.
//!
//! - **c16** : complex128 `(4,L,L,L,L,2,2)`, C-order, IDENTIQUE à l'octet à `heritage::su2_hb::write_npy_quat`
//!   (même en-tête, même disposition), donc lisible par numpy et par `read_npy_quat` ;
//! - **q64** : float64 `(4,L,L,L,L,4)`, quaternions (a0, a1, a2, a3), SCALAIRE EN PREMIER (U = a0 + i(a1σ1 + a2σ2 + a3σ3),
//!   comme su2_hb, ymformat et `ChampHote`), moitié de la place ;
//! - tableaux f64, f32 et u32 de forme quelconque ;
//! - la lecture accepte `.npy` et `.npy.zst` (décompression à la volée par `zstd -dc`) ;
//! - tout champ lu ou écrit doit être SU(2) à 1e-12 (|q| = 1, structure exacte pour le c16) ;
//! - les conversions c16 ↔ q64 se font en flux (mémoire bornée) et sont EXACTES : on vérifie par
//!   l'aller-retour que la conversion inverse redonne le fichier d'origine à l'octet.

use super::compression::{self, FluxZstd};
use super::garde::GardeSsd;
use crate::commun::{ChampHote, Erreur, Resultat};
use crate::heritage::manifest::{hex, Sha256};
use crate::heritage::su2_hb::{mat_to_quat, qnorm, quat_to_mat};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Tolérance SU(2) à la lecture et à l'écriture (même règle que `su2_hb::read_npy_quat`).
pub const TOLERANCE_SU2: f64 = 1e-12;
/// Liens traités par bloc dans les flux (c16 : 2 Mio par bloc de 32768 liens).
const LIENS_PAR_BLOC: usize = 1 << 15;
/// Éléments traités par bloc pour les tableaux.
const ELEMENTS_PAR_BLOC: usize = 1 << 18;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FormatChamp {
    /// complex128 (4,L,L,L,L,2,2), C-order : identique à l'octet à dump_npy de su2_hip.cpp.
    C16,
    /// float64 (4,L,L,L,L,4) quaternions (a0, a1, a2, a3), scalaire en premier : moitié de la taille, format interne
    /// de ym4_noyaux.
    Q64,
}

impl FormatChamp {
    pub fn nom(self) -> &'static str {
        match self {
            FormatChamp::C16 => "c16",
            FormatChamp::Q64 => "q64",
        }
    }

    /// Octets par lien dans le fichier : 64 (c16) ou 32 (q64).
    pub fn octets_par_lien(self) -> usize {
        match self {
            FormatChamp::C16 => 64,
            FormatChamp::Q64 => 32,
        }
    }
}

impl FromStr for FormatChamp {
    type Err = Erreur;
    fn from_str(s: &str) -> Resultat<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "c16" => Ok(FormatChamp::C16),
            "q64" => Ok(FormatChamp::Q64),
            autre => Err(Erreur::Config(format!("format de champ inconnu « {autre} » (c16|q64)"))),
        }
    }
}

// ------------------------------------------------------------------------------------------
// En-tête NPY
// ------------------------------------------------------------------------------------------

/// En-tête NPY v1.0 décodé.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnteteNpy {
    /// Type, ex. « <c16 », « <f8 ».
    pub descr: String,
    pub fortran: bool,
    pub forme: Vec<usize>,
    /// Taille de l'en-tête (10 + longueur du dictionnaire), donc offset des données.
    pub octets_entete: usize,
}

impl EnteteNpy {
    /// Nombre d'éléments (produit de la forme, 1 pour une forme vide). None en cas de dépassement.
    pub fn nelements(&self) -> Option<usize> {
        self.forme.iter().try_fold(1usize, |a, &d| a.checked_mul(d))
    }
}

/// Texte de la forme, comme numpy : `()`, `(n,)`, `(a, b, c)`.
fn texte_forme(forme: &[usize]) -> String {
    match forme {
        [] => "()".to_string(),
        [n] => format!("({n},)"),
        _ => {
            let v: Vec<String> = forme.iter().map(|d| d.to_string()).collect();
            format!("({})", v.join(", "))
        }
    }
}

/// En-tête NPY v1.0 d'un tableau C-order : dictionnaire complété d'espaces puis `\n`, de sorte que
/// 10 + longueur soit un multiple de 64 (même calcul que `su2_hb::npy_header` et numpy).
pub fn entete_tableau(descr: &str, forme: &[usize]) -> Vec<u8> {
    let mut hd = format!(
        "{{'descr': '{descr}', 'fortran_order': False, 'shape': {}, }}",
        texte_forme(forme)
    );
    let tot = 10 + hd.len() + 1;
    let pad = (64 - tot % 64) % 64;
    hd.push_str(&" ".repeat(pad));
    hd.push('\n');
    let mut out = vec![0x93u8, b'N', b'U', b'M', b'P', b'Y', 1, 0];
    out.extend_from_slice(&(hd.len() as u16).to_le_bytes());
    out.extend_from_slice(hd.as_bytes());
    out
}

/// En-tête d'un champ de lien L⁴ au format demandé.
pub fn entete_champ(l: usize, format: FormatChamp) -> Vec<u8> {
    match format {
        FormatChamp::C16 => entete_tableau("<c16", &[4, l, l, l, l, 2, 2]),
        FormatChamp::Q64 => entete_tableau("<f8", &[4, l, l, l, l, 4]),
    }
}

/// Taille totale (octets) du fichier d'un champ de côté `l` : en-tête + 4·L⁴ liens (c16 : 256·L⁴ ; q64 : 128·L⁴).
/// Saturante.
pub fn octets_champ(l: usize, format: FormatChamp) -> u64 {
    let nliens = 4u64.saturating_mul((l as u64).saturating_pow(4));
    let donnees = nliens.saturating_mul(format.octets_par_lien() as u64);
    donnees.saturating_add(entete_champ(l, format).len() as u64)
}

/// Valeur textuelle qui suit `'cle':` dans le dictionnaire de l'en-tête.
fn valeur_apres<'a>(texte: &'a str, cle: &str) -> Option<&'a str> {
    let motif = format!("'{cle}':");
    let i = texte.find(&motif)? + motif.len();
    Some(texte[i..].trim_start())
}

/// Lit et décode un en-tête NPY v1.0 (le flux est positionné sur les données ensuite).
pub fn lire_entete<R: Read>(r: &mut R) -> Resultat<EnteteNpy> {
    let mut pre = [0u8; 10];
    lire_exact(r, &mut pre, "en-tête NPY")?;
    if &pre[..6] != b"\x93NUMPY" {
        return Err(Erreur::Format("ce n'est pas un fichier NPY (signature absente)".into()));
    }
    if pre[6] != 1 {
        return Err(Erreur::Format(format!(
            "NPY version {}.{} non gérée (v1.x seulement)",
            pre[6], pre[7]
        )));
    }
    let hl = u16::from_le_bytes([pre[8], pre[9]]) as usize;
    let mut brut = vec![0u8; hl];
    lire_exact(r, &mut brut, "dictionnaire d'en-tête NPY")?;
    let texte = String::from_utf8_lossy(&brut).into_owned();

    let descr = {
        let v = valeur_apres(&texte, "descr").ok_or_else(|| Erreur::Format("en-tête NPY : 'descr' absent".into()))?;
        let v = v
            .strip_prefix('\'')
            .ok_or_else(|| Erreur::Format("en-tête NPY : 'descr' mal formé".into()))?;
        v[..v
            .find('\'')
            .ok_or_else(|| Erreur::Format("en-tête NPY : 'descr' non terminé".into()))?]
            .to_string()
    };
    let fortran = {
        let v = valeur_apres(&texte, "fortran_order")
            .ok_or_else(|| Erreur::Format("en-tête NPY : 'fortran_order' absent".into()))?;
        if v.starts_with("True") {
            true
        } else if v.starts_with("False") {
            false
        } else {
            return Err(Erreur::Format("en-tête NPY : 'fortran_order' mal formé".into()));
        }
    };
    let forme = {
        let v = valeur_apres(&texte, "shape").ok_or_else(|| Erreur::Format("en-tête NPY : 'shape' absent".into()))?;
        let v = v
            .strip_prefix('(')
            .ok_or_else(|| Erreur::Format("en-tête NPY : 'shape' mal formé".into()))?;
        let fin = v
            .find(')')
            .ok_or_else(|| Erreur::Format("en-tête NPY : 'shape' non terminé".into()))?;
        let mut dims = Vec::new();
        for t in v[..fin].split(',').map(str::trim).filter(|t| !t.is_empty()) {
            dims.push(
                t.parse::<usize>()
                    .map_err(|_| Erreur::Format(format!("en-tête NPY : dimension illisible « {t} »")))?,
            );
        }
        dims
    };
    Ok(EnteteNpy {
        descr,
        fortran,
        forme,
        octets_entete: 10 + hl,
    })
}

/// `read_exact` avec un message qui distingue un fichier tronqué d'une erreur d'E/S.
pub(crate) fn lire_exact<R: Read>(r: &mut R, tampon: &mut [u8], quoi: &str) -> Resultat<()> {
    r.read_exact(tampon).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            Erreur::Format(format!("fichier tronqué ({quoi})"))
        } else {
            Erreur::io(format!("lecture ({quoi})"), e)
        }
    })
}

/// Disposition d'un champ SU(2) d'après son en-tête : (format, L). Refuse tout autre en-tête.
pub fn disposition_champ(e: &EnteteNpy) -> Resultat<(FormatChamp, usize)> {
    if e.fortran {
        return Err(Erreur::Format("fortran_order True refusé (C-order attendu)".into()));
    }
    let d = &e.forme;
    let cubique = |d: &[usize]| d.len() >= 5 && d[1] == d[2] && d[1] == d[3] && d[1] == d[4];
    match e.descr.as_str() {
        "<c16" if d.len() == 7 && d[0] == 4 && d[5] == 2 && d[6] == 2 && cubique(d) => Ok((FormatChamp::C16, d[1])),
        "<f8" if d.len() == 6 && d[0] == 4 && d[5] == 4 && cubique(d) => Ok((FormatChamp::Q64, d[1])),
        _ => Err(Erreur::Format(format!(
            "en-tête de champ refusé : descr {} forme {:?} (attendu <c16 (4,L,L,L,L,2,2) ou <f8 (4,L,L,L,L,4))",
            e.descr, e.forme
        ))),
    }
}

// ------------------------------------------------------------------------------------------
// Lecture : fichier ou flux zstd, avec hachage facultatif
// ------------------------------------------------------------------------------------------

/// Source de lecture d'un `.npy` ou d'un `.npy.zst`.
enum Lecture {
    Fichier(BufReader<File>),
    Zstd(FluxZstd),
}

impl Read for Lecture {
    fn read(&mut self, t: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Lecture::Fichier(f) => f.read(t),
            Lecture::Zstd(z) => z.read(t),
        }
    }
}

impl Lecture {
    fn ouvrir(chemin: &Path) -> Resultat<Lecture> {
        if compression::est_zst(chemin) {
            Ok(Lecture::Zstd(compression::ouvrir_flux(chemin)?))
        } else {
            let f = File::open(chemin).map_err(|e| Erreur::io(format!("ouverture {}", chemin.display()), e))?;
            Ok(Lecture::Fichier(BufReader::with_capacity(1 << 20, f)))
        }
    }

    /// Fin de lecture : pour zstd, vérifie le code de retour (détecte troncature et corruption).
    fn terminer(self) -> Resultat<()> {
        match self {
            Lecture::Fichier(_) => Ok(()),
            Lecture::Zstd(z) => z.terminer(),
        }
    }
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

/// Ajoute le chemin au message d'une erreur de format.
fn avec_chemin(e: Erreur, chemin: &Path) -> Erreur {
    match e {
        Erreur::Format(m) => Erreur::Format(format!("{} : {m}", chemin.display())),
        autre => autre,
    }
}

fn f64_le(b: &[u8]) -> f64 {
    f64::from_le_bytes(b[..8].try_into().expect("8 octets"))
}

/// Décode UN lien (c16 : 64 octets ; q64 : 32 octets) en quaternion : `Ok(q)` s'il est SU(2) à 1e-12 (composantes finies,
/// | |q| − 1 | ≤ 1e-12 et, en c16, structure exacte `M = quat_to_mat(q)` à 1e-12), sinon `Err(défaut)` (tel que calculé : NaN ou ∞
/// pour un lien non fini).
/// Règle unique de la lecture des champs (`lire_champ`) et des bords lus (`su2 boite --bord-npy`).
pub(crate) fn decoder_lien(format: FormatChamp, lien: &[u8]) -> Result<[f64; 4], f64> {
    let (q, defaut) = match format {
        FormatChamp::C16 => {
            let mut m = [0f64; 8];
            for (k, v) in m.iter_mut().enumerate() {
                *v = f64_le(&lien[8 * k..]);
            }
            let q = mat_to_quat(&m);
            let retour = quat_to_mat(q);
            let ecart = m
                .iter()
                .zip(retour.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max);
            (q, ecart.max((qnorm(q) - 1.0).abs()))
        }
        FormatChamp::Q64 => {
            let q = [
                f64_le(lien),
                f64_le(&lien[8..]),
                f64_le(&lien[16..]),
                f64_le(&lien[24..]),
            ];
            (q, (qnorm(q) - 1.0).abs())
        }
    };
    if !q.iter().all(|x| x.is_finite()) || defaut.is_nan() || defaut > TOLERANCE_SU2 {
        return Err(defaut);
    }
    Ok(q)
}

/// Décode un bloc de liens (c16 ou q64) en quaternions, en refusant tout lien hors SU(2) à 1e-12.
/// `premier` = indice du premier lien du bloc (pour les messages).
fn decoder_bloc(format: FormatChamp, octets: &[u8], premier: usize, sortie: &mut Vec<f64>) -> Resultat<()> {
    let taille = format.octets_par_lien();
    for (i, lien) in octets.chunks_exact(taille).enumerate() {
        match decoder_lien(format, lien) {
            Ok(q) => sortie.extend_from_slice(&q),
            Err(defaut) => {
                return Err(Erreur::Format(format!(
                    "lien {} hors SU(2) : défaut {defaut:e} (tolérance 1e-12)",
                    premier + i
                )));
            }
        }
    }
    Ok(())
}

/// Encode des quaternions (4 réels par lien) dans le format demandé, en ajoutant à `sortie`.
fn encoder_bloc(format: FormatChamp, quats: &[f64], sortie: &mut Vec<u8>) {
    match format {
        FormatChamp::C16 => {
            for q in quats.chunks_exact(4) {
                for v in quat_to_mat([q[0], q[1], q[2], q[3]]) {
                    sortie.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        FormatChamp::Q64 => {
            for v in quats {
                sortie.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
}

/// Nombre de liens d'un champ de côté `l` (4·L⁴), ou erreur si cela déborde.
fn nliens(l: usize) -> Resultat<usize> {
    l.checked_pow(4)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| Erreur::Format(format!("L = {l} : taille de champ hors limites")))
}

/// Parcourt les liens d'un flux positionné sur les données : chaque bloc décodé et validé est passé à `f`.
/// Refuse un fichier tronqué ou suivi d'octets en trop.
fn parcourir_liens<R: Read>(
    r: &mut R,
    format: FormatChamp,
    l: usize,
    mut f: impl FnMut(&[f64]) -> Resultat<()>,
) -> Resultat<()> {
    let total = nliens(l)?;
    let taille = format.octets_par_lien();
    let mut tampon = vec![0u8; LIENS_PAR_BLOC.min(total) * taille];
    let mut quats = Vec::with_capacity(4 * LIENS_PAR_BLOC.min(total));
    let mut fait = 0usize;
    while fait < total {
        let n = LIENS_PAR_BLOC.min(total - fait);
        lire_exact(r, &mut tampon[..n * taille], &format!("lien {fait} sur {total}"))?;
        quats.clear();
        decoder_bloc(format, &tampon[..n * taille], fait, &mut quats)?;
        f(&quats)?;
        fait += n;
    }
    let mut un = [0u8; 1];
    match r.read(&mut un) {
        Ok(0) => Ok(()),
        Ok(_) => Err(Erreur::Format("octets en trop après les données du champ".into())),
        Err(e) => Err(Erreur::io("lecture de la fin du champ", e)),
    }
}

/// Vérifie, pour un fichier non compressé, que sa taille est exactement celle annoncée par l'en-tête.
fn verifier_taille(chemin: &Path, e: &EnteteNpy, octets_donnees: Option<u64>) -> Resultat<()> {
    if compression::est_zst(chemin) {
        return Ok(());
    }
    let Some(donnees) = octets_donnees else {
        return Err(Erreur::Format(format!(
            "{} : taille annoncée hors limites",
            chemin.display()
        )));
    };
    let reel = std::fs::metadata(chemin)
        .map_err(|er| Erreur::io(format!("stat {}", chemin.display()), er))?
        .len();
    let attendu = e.octets_entete as u64 + donnees;
    if reel != attendu {
        return Err(Erreur::Format(format!(
            "{} : taille {reel} octets au lieu de {attendu} annoncés",
            chemin.display()
        )));
    }
    Ok(())
}

/// Format et côté L d'un champ, d'après le seul en-tête (rapide, ne lit pas les données).
pub fn inspecter_champ(chemin: &Path) -> Resultat<(FormatChamp, usize)> {
    let mut flux = Lecture::ouvrir(chemin)?;
    let e = lire_entete(&mut flux).map_err(|er| avec_chemin(er, chemin))?;
    disposition_champ(&e).map_err(|er| avec_chemin(er, chemin))
}

/// Lit un champ (détection du format par l'en-tête ; .npy.zst décompressé à la volée).
/// Refuse une matrice hors SU(2) à 1e-12 (même règle que su2_hb::read_npy_quat).
pub fn lire_champ(chemin: &Path) -> Resultat<ChampHote> {
    let mut flux = Lecture::ouvrir(chemin)?;
    let e = lire_entete(&mut flux).map_err(|er| avec_chemin(er, chemin))?;
    let (format, l) = disposition_champ(&e).map_err(|er| avec_chemin(er, chemin))?;
    let total = nliens(l).map_err(|er| avec_chemin(er, chemin))?;
    verifier_taille(chemin, &e, (total as u64).checked_mul(format.octets_par_lien() as u64))?;
    let mut q = Vec::with_capacity((4 * total).min(1 << 24));
    parcourir_liens(&mut flux, format, l, |bloc| {
        q.extend_from_slice(bloc);
        Ok(())
    })
    .map_err(|er| avec_chemin(er, chemin))?;
    flux.terminer()?;
    Ok(ChampHote { l, q })
}

// ------------------------------------------------------------------------------------------
// Écriture atomique
// ------------------------------------------------------------------------------------------

/// Résultat d'une écriture : taille réelle et, si demandé, sha256 du contenu écrit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResumeEcriture {
    pub octets: u64,
    pub sha256: Option<String>,
}

/// Écriture atomique : `<nom>.part`, fsync, rename, fsync du répertoire. La garde SSD est consultée
/// AVANT la création du moindre fichier. Un `.part` jamais publié est supprimé à la destruction.
pub(crate) struct Ecriture {
    part: PathBuf,
    dest: PathBuf,
    w: Option<BufWriter<File>>,
    sha: Option<Sha256>,
    octets: u64,
    publie: bool,
}

impl Ecriture {
    pub(crate) fn creer(dest: &Path, octets_prevus: u64, garde: &GardeSsd, hacher: bool) -> Resultat<Ecriture> {
        garde.verifier(dest, octets_prevus)?;
        let mut s = dest.as_os_str().to_os_string();
        s.push(".part");
        let part = PathBuf::from(s);
        let f = File::create(&part).map_err(|e| Erreur::io(format!("création {}", part.display()), e))?;
        Ok(Ecriture {
            part,
            dest: dest.to_path_buf(),
            w: Some(BufWriter::with_capacity(1 << 22, f)),
            sha: hacher.then(Sha256::new),
            octets: 0,
            publie: false,
        })
    }

    pub(crate) fn ecrire(&mut self, b: &[u8]) -> Resultat<()> {
        let w = self.w.as_mut().expect("écriture déjà fermée");
        w.write_all(b)
            .map_err(|e| Erreur::io(format!("écriture {}", self.part.display()), e))?;
        if let Some(h) = self.sha.as_mut() {
            h.update(b);
        }
        self.octets += b.len() as u64;
        Ok(())
    }

    /// Vide le tampon et fait un fsync du fichier. Le `.part` reste en place (relisible pour vérification).
    pub(crate) fn fermer(&mut self) -> Resultat<ResumeEcriture> {
        let w = self.w.take().expect("écriture déjà fermée");
        let f = w
            .into_inner()
            .map_err(|e| Erreur::io(format!("vidage {}", self.part.display()), e.into_error()))?;
        f.sync_all()
            .map_err(|e| Erreur::io(format!("fsync {}", self.part.display()), e))?;
        let sha256 = self.sha.take().map(|h| hex(&h.finalize()));
        Ok(ResumeEcriture {
            octets: self.octets,
            sha256,
        })
    }

    /// Renomme le `.part` sur la destination puis fait un fsync du répertoire.
    pub(crate) fn publier(mut self) -> Resultat<()> {
        std::fs::rename(&self.part, &self.dest).map_err(|e| {
            Erreur::io(
                format!("renommage {} -> {}", self.part.display(), self.dest.display()),
                e,
            )
        })?;
        self.publie = true;
        compression::synchroniser_repertoire(&self.dest);
        Ok(())
    }
}

impl Drop for Ecriture {
    fn drop(&mut self) {
        if !self.publie {
            self.w.take();
            let _ = std::fs::remove_file(&self.part);
        }
    }
}

/// Valide un champ avant écriture : taille cohérente, tous les liens finis et unitaires à 1e-12.
fn valider_champ(c: &ChampHote) -> Resultat<()> {
    let n = nliens(c.l)?;
    if c.l == 0 || c.q.len() != 4 * n {
        return Err(Erreur::Config(format!(
            "champ incohérent : L = {} attend {} réels, reçu {}",
            c.l,
            4 * n,
            c.q.len()
        )));
    }
    for (i, q) in c.q.chunks_exact(4).enumerate() {
        let defaut = (qnorm([q[0], q[1], q[2], q[3]]) - 1.0).abs();
        if !q.iter().all(|x| x.is_finite()) || defaut.is_nan() || defaut > TOLERANCE_SU2 {
            return Err(Erreur::Validation(format!(
                "écriture refusée : lien {i} hors SU(2) (défaut {defaut:e} > 1e-12) ; un champ F32 se renormalise \
                 en double précision par stockage::npy::renormaliser avant l'écriture"
            )));
        }
    }
    Ok(())
}

/// Renormalise en double précision chaque quaternion à |q| = 1 (utile après un calcul F32).
/// Rend le plus grand défaut ||q| − 1| avant correction. Refuse un lien nul ou non fini.
pub fn renormaliser(c: &mut ChampHote) -> Resultat<f64> {
    let mut pire = 0.0f64;
    for (i, q) in c.q.chunks_exact_mut(4).enumerate() {
        let n = qnorm([q[0], q[1], q[2], q[3]]);
        if !n.is_finite() || n < 1e-300 {
            return Err(Erreur::Validation(format!(
                "lien {i} nul ou non fini : renormalisation impossible"
            )));
        }
        pire = pire.max((n - 1.0).abs());
        for v in q.iter_mut() {
            *v /= n;
        }
    }
    Ok(pire)
}

fn ecrire_champ_interne(
    chemin: &Path,
    c: &ChampHote,
    format: FormatChamp,
    garde: &GardeSsd,
    hacher: bool,
) -> Resultat<ResumeEcriture> {
    valider_champ(c)?;
    let entete = entete_champ(c.l, format);
    let prevu = entete.len() as u64 + (c.q.len() as u64 / 4) * format.octets_par_lien() as u64;
    let mut ec = Ecriture::creer(chemin, prevu, garde, hacher)?;
    ec.ecrire(&entete)?;
    let mut tampon = Vec::with_capacity(LIENS_PAR_BLOC * format.octets_par_lien());
    for bloc in c.q.chunks(4 * LIENS_PAR_BLOC) {
        tampon.clear();
        encoder_bloc(format, bloc, &mut tampon);
        ec.ecrire(&tampon)?;
    }
    let r = ec.fermer()?;
    ec.publier()?;
    Ok(r)
}

/// Écrit le champ au format demandé (atomique, garde SSD avant le premier octet).
/// Refuse un champ hors SU(2) à 1e-12 : tout ce qui s'écrit se relit.
pub fn ecrire_champ(chemin: &Path, c: &ChampHote, format: FormatChamp, garde: &GardeSsd) -> Resultat<()> {
    ecrire_champ_interne(chemin, c, format, garde, false).map(|_| ())
}

/// Comme [`ecrire_champ`], et rend le sha256 du fichier produit (calculé pendant l'écriture : pas de relecture
/// pour le manifeste, voir `Manifeste::fichier_connu`).
pub fn ecrire_champ_hache(
    chemin: &Path,
    c: &ChampHote,
    format: FormatChamp,
    garde: &GardeSsd,
) -> Resultat<ResumeEcriture> {
    ecrire_champ_interne(chemin, c, format, garde, true)
}

// ------------------------------------------------------------------------------------------
// Conversions c16 <-> q64 en flux, avec vérification de l'aller-retour
// ------------------------------------------------------------------------------------------

/// sha256 du fichier qu'on obtiendrait en réencodant en `format_src` (en-tête canonique compris) le champ
/// contenu dans `chemin_converti` (au format `format_converti`).
fn sha_reencode(
    chemin_converti: &Path,
    format_converti: FormatChamp,
    format_src: FormatChamp,
    l: usize,
) -> Resultat<String> {
    let f =
        File::open(chemin_converti).map_err(|e| Erreur::io(format!("relecture {}", chemin_converti.display()), e))?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let e = lire_entete(&mut r).map_err(|er| avec_chemin(er, chemin_converti))?;
    let (fmt, l2) = disposition_champ(&e).map_err(|er| avec_chemin(er, chemin_converti))?;
    if fmt != format_converti || l2 != l {
        return Err(Erreur::Validation(
            "fichier converti : en-tête inattendu à la relecture".into(),
        ));
    }
    let mut h = Sha256::new();
    h.update(&entete_champ(l, format_src));
    let mut tampon = Vec::new();
    parcourir_liens(&mut r, format_converti, l, |bloc| {
        tampon.clear();
        encoder_bloc(format_src, bloc, &mut tampon);
        h.update(&tampon);
        Ok(())
    })
    .map_err(|er| avec_chemin(er, chemin_converti))?;
    Ok(hex(&h.finalize()))
}

/// Convertit un champ `src` (c16 ou q64, `.npy` ou `.npy.zst`) vers le format `cible`, en flux (mémoire bornée),
/// dans `dst` (écriture atomique ; `dst` peut être `src` : le remplacement est un renommage).
///
/// Si `verifier` : avant de publier, on relit le fichier converti, on le réencode dans le format d'origine
/// et on exige le MÊME sha256 que le fichier d'origine (aller-retour exact à l'octet). Sinon le `.part` est
/// supprimé et l'erreur `Validation` rendue : l'original n'a pas été touché.
pub fn convertir_champ(
    src: &Path,
    dst: &Path,
    cible: FormatChamp,
    garde: &GardeSsd,
    verifier: bool,
) -> Resultat<ResumeEcriture> {
    convertir_champ_avec(src, dst, cible, garde, verifier, &mut |_| Ok(()))
}

/// Comme [`convertir_champ`] ; `avant_publier` est appelé APRÈS la vérification et AVANT le renommage final,
/// avec la taille et le sha256 du fichier converti (le journal d'éclaircissement y inscrit son intention :
/// un arrêt brutal ne laisse jamais un fichier modifié sans trace). Une erreur de `avant_publier` annule la
/// publication (l'original reste intact).
pub fn convertir_champ_avec(
    src: &Path,
    dst: &Path,
    cible: FormatChamp,
    garde: &GardeSsd,
    verifier: bool,
    avant_publier: &mut dyn FnMut(&ResumeEcriture) -> Resultat<()>,
) -> Resultat<ResumeEcriture> {
    let mut flux = LecteurHache {
        r: Lecture::ouvrir(src)?,
        h: Sha256::new(),
    };
    let e = lire_entete(&mut flux).map_err(|er| avec_chemin(er, src))?;
    let (format_src, l) = disposition_champ(&e).map_err(|er| avec_chemin(er, src))?;
    if format_src == cible {
        return Err(Erreur::Config(format!(
            "{} est déjà au format {}",
            src.display(),
            cible.nom()
        )));
    }
    let mut ec = Ecriture::creer(dst, octets_champ(l, cible), garde, true)?;
    ec.ecrire(&entete_champ(l, cible))?;
    let mut tampon = Vec::with_capacity(LIENS_PAR_BLOC * cible.octets_par_lien());
    parcourir_liens(&mut flux, format_src, l, |bloc| {
        tampon.clear();
        encoder_bloc(cible, bloc, &mut tampon);
        ec.ecrire(&tampon)
    })
    .map_err(|er| avec_chemin(er, src))?;
    let LecteurHache { r, h } = flux;
    r.terminer()?;
    let sha_src = hex(&h.finalize());
    let resume = ec.fermer()?;
    if verifier {
        let retour = sha_reencode(&ec.part, cible, format_src, l)?;
        if retour != sha_src {
            return Err(Erreur::Validation(format!(
                "conversion {} -> {} de {} : l'aller-retour n'est PAS exact (sha256 {retour} au lieu de {sha_src}) ; \
                 original conservé",
                format_src.nom(),
                cible.nom(),
                src.display()
            )));
        }
    }
    avant_publier(&resume)?;
    ec.publier()?;
    Ok(resume)
}

// ------------------------------------------------------------------------------------------
// Tableaux f64 / f32 / u32 de forme quelconque
// ------------------------------------------------------------------------------------------

/// Éléments écrivables dans un NPY.
trait Element: Copy {
    const DESCR: &'static str;
    fn ajouter(self, sortie: &mut Vec<u8>);
}

impl Element for f64 {
    const DESCR: &'static str = "<f8";
    fn ajouter(self, s: &mut Vec<u8>) {
        s.extend_from_slice(&self.to_le_bytes());
    }
}
impl Element for f32 {
    const DESCR: &'static str = "<f4";
    fn ajouter(self, s: &mut Vec<u8>) {
        s.extend_from_slice(&self.to_le_bytes());
    }
}
impl Element for u32 {
    const DESCR: &'static str = "<u4";
    fn ajouter(self, s: &mut Vec<u8>) {
        s.extend_from_slice(&self.to_le_bytes());
    }
}

fn ecrire_tableau<T: Element>(chemin: &Path, forme: &[usize], donnees: &[T], garde: &GardeSsd) -> Resultat<()> {
    let attendu = forme.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
    if attendu != Some(donnees.len()) {
        return Err(Erreur::Config(format!(
            "forme {forme:?} incompatible avec {} valeurs fournies pour {}",
            donnees.len(),
            chemin.display()
        )));
    }
    let entete = entete_tableau(T::DESCR, forme);
    let taille = std::mem::size_of::<T>() as u64;
    let mut ec = Ecriture::creer(
        chemin,
        entete.len() as u64 + donnees.len() as u64 * taille,
        garde,
        false,
    )?;
    ec.ecrire(&entete)?;
    let mut tampon = Vec::with_capacity(ELEMENTS_PAR_BLOC * std::mem::size_of::<T>());
    for bloc in donnees.chunks(ELEMENTS_PAR_BLOC) {
        tampon.clear();
        for &v in bloc {
            v.ajouter(&mut tampon);
        }
        ec.ecrire(&tampon)?;
    }
    ec.fermer()?;
    ec.publier()
}

/// Tableau f64 de forme quelconque (C-order).
pub fn ecrire_f64(chemin: &Path, forme: &[usize], donnees: &[f64], garde: &GardeSsd) -> Resultat<()> {
    ecrire_tableau(chemin, forme, donnees, garde)
}

pub fn ecrire_f32(chemin: &Path, forme: &[usize], donnees: &[f32], garde: &GardeSsd) -> Resultat<()> {
    ecrire_tableau(chemin, forme, donnees, garde)
}

/// Tableau u32 de forme quelconque (C-order).
pub fn ecrire_u32(chemin: &Path, forme: &[usize], donnees: &[u32], garde: &GardeSsd) -> Resultat<()> {
    ecrire_tableau(chemin, forme, donnees, garde)
}

/// Lit un tableau dont le type est l'un des `(descr, octets)` acceptés ; chaque élément est décodé par `decode`.
fn lire_tableau<T>(
    chemin: &Path,
    acceptes: &[(&str, usize)],
    decode: fn(&[u8]) -> T,
) -> Resultat<(Vec<usize>, Vec<T>)> {
    let mut flux = Lecture::ouvrir(chemin)?;
    let e = lire_entete(&mut flux).map_err(|er| avec_chemin(er, chemin))?;
    if e.fortran {
        return Err(Erreur::Format(format!(
            "{} : fortran_order True refusé",
            chemin.display()
        )));
    }
    let Some(&(_, taille)) = acceptes.iter().find(|(d, _)| *d == e.descr) else {
        let noms: Vec<&str> = acceptes.iter().map(|(d, _)| *d).collect();
        return Err(Erreur::Format(format!(
            "{} : type {} refusé (attendu {})",
            chemin.display(),
            e.descr,
            noms.join(" ou ")
        )));
    };
    let n = e
        .nelements()
        .ok_or_else(|| Erreur::Format(format!("{} : forme hors limites", chemin.display())))?;
    verifier_taille(chemin, &e, (n as u64).checked_mul(taille as u64))?;
    let mut sortie: Vec<T> = Vec::with_capacity(n.min(1 << 24));
    let mut tampon = vec![0u8; ELEMENTS_PAR_BLOC.min(n.max(1)) * taille];
    let mut fait = 0usize;
    while fait < n {
        let m = ELEMENTS_PAR_BLOC.min(n - fait);
        lire_exact(&mut flux, &mut tampon[..m * taille], &format!("élément {fait} sur {n}"))
            .map_err(|er| avec_chemin(er, chemin))?;
        sortie.extend(tampon[..m * taille].chunks_exact(taille).map(decode));
        fait += m;
    }
    let mut un = [0u8; 1];
    match flux.read(&mut un) {
        Ok(0) => {}
        Ok(_) => {
            return Err(Erreur::Format(format!(
                "{} : octets en trop après les données",
                chemin.display()
            )))
        }
        Err(er) => return Err(Erreur::io(format!("lecture {}", chemin.display()), er)),
    }
    flux.terminer()?;
    Ok((e.forme, sortie))
}

/// Lit un tableau float64 (ou float32, promu sans perte) : (forme, valeurs en C-order).
pub fn lire_f64(chemin: &Path) -> Resultat<(Vec<usize>, Vec<f64>)> {
    lire_tableau(chemin, &[("<f8", 8), ("<f4", 4)], |b| match b.len() {
        8 => f64_le(b),
        _ => f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
    })
}

/// Lit un tableau float32 strict.
pub fn lire_f32(chemin: &Path) -> Resultat<(Vec<usize>, Vec<f32>)> {
    lire_tableau(chemin, &[("<f4", 4)], |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Lit un tableau uint32 strict.
pub fn lire_u32(chemin: &Path) -> Resultat<(Vec<usize>, Vec<u32>)> {
    lire_tableau(chemin, &[("<u4", 4)], |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heritage::su2_hb;

    #[test]
    fn entete_c16_identique_a_su2_hb() {
        for l in [2usize, 4, 6, 8, 16, 32, 64, 100] {
            assert_eq!(entete_champ(l, FormatChamp::C16), su2_hb::npy_header(l), "L = {l}");
        }
    }

    #[test]
    fn entetes_alignes_sur_64() {
        for forme in [
            vec![],
            vec![7],
            vec![3, 5],
            vec![4, 8, 8, 8, 8, 4],
            vec![123456789, 2, 3],
        ] {
            for d in ["<f8", "<f4", "<u4", "<c16"] {
                let h = entete_tableau(d, &forme);
                assert_eq!(h.len() % 64, 0, "{d} {forme:?}");
                assert_eq!(*h.last().unwrap(), b'\n');
                let e = lire_entete(&mut &h[..]).unwrap();
                assert_eq!((e.descr.as_str(), &e.forme), (d, &forme));
                assert_eq!(e.octets_entete, h.len());
            }
        }
    }

    #[test]
    fn forme_ecrite_comme_numpy() {
        assert_eq!(texte_forme(&[]), "()");
        assert_eq!(texte_forme(&[5]), "(5,)");
        assert_eq!(texte_forme(&[2, 3]), "(2, 3)");
    }

    #[test]
    fn entetes_refuses() {
        assert!(lire_entete(&mut &b"pas du npy"[..]).is_err());
        let mut v2 = entete_tableau("<f8", &[1]);
        v2[6] = 2;
        assert!(lire_entete(&mut &v2[..]).is_err());
        let mut tronque = entete_tableau("<f8", &[1]);
        tronque.truncate(30);
        assert!(lire_entete(&mut &tronque[..]).is_err());
    }

    #[test]
    fn taille_des_fichiers_de_champ() {
        // c16 = 256 L^4 + en-tête ; q64 = 128 L^4 + en-tête
        assert_eq!(
            octets_champ(8, FormatChamp::C16),
            256 * 4096 + su2_hb::npy_header(8).len() as u64
        );
        assert_eq!(
            octets_champ(8, FormatChamp::Q64),
            128 * 4096 + entete_champ(8, FormatChamp::Q64).len() as u64
        );
        assert_eq!(octets_champ(usize::MAX, FormatChamp::C16), u64::MAX);
    }

    #[test]
    fn format_depuis_texte() {
        assert_eq!("C16".parse::<FormatChamp>().unwrap(), FormatChamp::C16);
        assert_eq!("q64".parse::<FormatChamp>().unwrap().nom(), "q64");
        assert!("f32".parse::<FormatChamp>().is_err());
    }

    #[test]
    fn decodage_refuse_nan_et_non_unitaire() {
        let mut sortie = Vec::new();
        let mut q = Vec::new();
        encoder_bloc(FormatChamp::Q64, &[1.0, 0.0, 0.0, 0.0], &mut q);
        assert!(decoder_bloc(FormatChamp::Q64, &q, 0, &mut sortie).is_ok());
        q.clear();
        encoder_bloc(FormatChamp::Q64, &[f64::NAN, 0.0, 0.0, 0.0], &mut q);
        assert!(decoder_bloc(FormatChamp::Q64, &q, 0, &mut sortie).is_err());
        q.clear();
        encoder_bloc(FormatChamp::Q64, &[1.0, 1e-5, 0.0, 0.0], &mut q);
        assert!(decoder_bloc(FormatChamp::Q64, &q, 0, &mut sortie).is_err());
    }

    #[test]
    fn renormalisation() {
        let mut c = ChampHote::froid(2).unwrap();
        for v in c.q.iter_mut() {
            *v *= 1.0 + 3e-7;
        }
        let pire = renormaliser(&mut c).unwrap();
        assert!(pire > 2e-7 && pire < 4e-7);
        assert!(c.defaut_unitarite() < 1e-15);
        c.q[0..4].copy_from_slice(&[0.0; 4]);
        assert!(renormaliser(&mut c).is_err());
    }
}
