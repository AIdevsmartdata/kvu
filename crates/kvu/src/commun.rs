//! Types partagés par tous les modules du kevinotron unifié.
//!
//! GELÉ (architecte) : toute modification passe par l'architecte, car chaque tâche en dépend.

use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

// ------------------------------------------------------------------------------------------
// Erreurs et codes de sortie
// ------------------------------------------------------------------------------------------

/// Erreur unique du crate. Chaque variante a un code de sortie stable (voir [`Erreur::code_sortie`]).
#[derive(Debug)]
pub enum Erreur {
    /// Entrée/sortie, avec le contexte (chemin, opération).
    Io { contexte: String, source: std::io::Error },
    /// Backend GPU absent ou appel runtime en échec.
    Gpu(String),
    /// Échec de compilation hiprtc/NVRTC (journal du compilateur complet).
    Compilation { backend: BackendKind, journal: String },
    /// Usage, option ou configuration invalide.
    Config(String),
    /// Ressources refusées (RAM, charge, moniteur en alerte) : rien n'est tué, on refuse.
    Ressources(String),
    /// Verrou GPU non obtenu dans le délai.
    Verrou(String),
    /// Garde SSD : l'écriture ferait passer l'espace libre sous le seuil.
    SsdPlein { chemin: PathBuf, libre_go: f64, besoin_go: f64, seuil_go: f64 },
    /// Format de fichier inattendu (NPY, JSON, CSV).
    Format(String),
    /// Gate ou contrôle de validation ROUGE.
    Validation(String),
    /// Fonction du squelette pas encore livrée par sa tâche.
    NonImplemente(&'static str),
}

/// Codes de sortie du binaire (stables, documentés dans GUIDE.md).
pub mod code {
    pub const OK: i32 = 0;
    pub const ECHEC: i32 = 1;
    pub const USAGE: i32 = 2;
    pub const GPU_INDISPONIBLE: i32 = 3;
    pub const IO: i32 = 4;
    pub const RESSOURCES: i32 = 5;
    pub const VERROU: i32 = 6;
    pub const SSD: i32 = 7;
    pub const VALIDATION: i32 = 8;
    pub const NON_IMPLEMENTE: i32 = 9;
}

impl Erreur {
    pub fn io(contexte: impl Into<String>, source: std::io::Error) -> Self {
        Erreur::Io { contexte: contexte.into(), source }
    }

    pub fn code_sortie(&self) -> i32 {
        match self {
            Erreur::Io { .. } | Erreur::Format(_) => code::IO,
            Erreur::Gpu(_) | Erreur::Compilation { .. } => code::GPU_INDISPONIBLE,
            Erreur::Config(_) => code::USAGE,
            Erreur::Ressources(_) => code::RESSOURCES,
            Erreur::Verrou(_) => code::VERROU,
            Erreur::SsdPlein { .. } => code::SSD,
            Erreur::Validation(_) => code::VALIDATION,
            Erreur::NonImplemente(_) => code::NON_IMPLEMENTE,
        }
    }
}

impl fmt::Display for Erreur {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Erreur::Io { contexte, source } => write!(f, "E/S ({contexte}) : {source}"),
            Erreur::Gpu(m) => write!(f, "GPU : {m}"),
            Erreur::Compilation { backend, journal } => {
                write!(f, "compilation {} en échec :\n{journal}", backend.nom())
            }
            Erreur::Config(m) => write!(f, "configuration : {m}"),
            Erreur::Ressources(m) => write!(f, "ressources refusées : {m}"),
            Erreur::Verrou(m) => write!(f, "verrou GPU : {m}"),
            Erreur::SsdPlein { chemin, libre_go, besoin_go, seuil_go } => write!(
                f,
                "garde SSD : {} a {libre_go:.1} Go libres, l'écriture demande {besoin_go:.2} Go, seuil {seuil_go:.0} Go : REFUS",
                chemin.display()
            ),
            Erreur::Format(m) => write!(f, "format : {m}"),
            Erreur::Validation(m) => write!(f, "validation ROUGE : {m}"),
            Erreur::NonImplemente(m) => write!(f, "non implémenté (squelette) : {m}"),
        }
    }
}

impl std::error::Error for Erreur {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Erreur::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub type Resultat<T> = Result<T, Erreur>;

// ------------------------------------------------------------------------------------------
// Machine, backend, précision
// ------------------------------------------------------------------------------------------

/// Backend de calcul effectivement utilisé.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    /// AMD via hiprtc (architecture `gfxNNNN` lue sur le périphérique). Toujours compilé (dlopen).
    Hip,
    /// NVIDIA via NVRTC + driver (architecture `sm_XY` lue sur le périphérique). Feature `cuda`.
    Cuda,
    /// Référence CPU (rayon, FP64).
    Cpu,
}

impl BackendKind {
    pub fn nom(self) -> &'static str {
        match self {
            BackendKind::Hip => "hip",
            BackendKind::Cuda => "cuda",
            BackendKind::Cpu => "cpu",
        }
    }
}

impl FromStr for BackendKind {
    type Err = Erreur;
    fn from_str(s: &str) -> Resultat<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "hip" => Ok(BackendKind::Hip),
            "cuda" => Ok(BackendKind::Cuda),
            "cpu" => Ok(BackendKind::Cpu),
            autre => Err(Erreur::Config(format!("backend inconnu « {autre} » (hip|cuda|cpu)"))),
        }
    }
}

/// Nom d'un profil de machine, mis en minuscules À LA COMPILATION depuis l'identifiant de la variante : le texte
/// n'est écrit nulle part dans le code, il vient du type (hôtes, chemins et seuils viennent de `kvu.toml`).
macro_rules! nom_de_variante {
    ($id:ident) => {{
        const LONGUEUR: usize = stringify!($id).len();
        const OCTETS: [u8; LONGUEUR] = {
            let source = stringify!($id).as_bytes();
            let mut sortie = [0u8; LONGUEUR];
            let mut i = 0;
            while i < LONGUEUR {
                sortie[i] = source[i].to_ascii_lowercase();
                i += 1;
            }
            sortie
        };
        match core::str::from_utf8(&OCTETS) {
            Ok(nom) => nom,
            Err(_) => panic!("identifiant de variante non UTF-8"),
        }
    }};
}

/// Nom du profil `Machine::Acemagic` (clé `[profils.<nom>]` de `kvu.toml`).
pub const NOM_PROFIL_ACEMAGIC: &str = nom_de_variante!(Acemagic);
/// Nom du profil `Machine::Gamer`.
pub const NOM_PROFIL_GAMER: &str = nom_de_variante!(Gamer);
/// Nom du profil `Machine::Inconnue`.
pub const NOM_PROFIL_INCONNUE: &str = nom_de_variante!(Inconnue);
/// Nom du profil `Machine::Location`.
pub const NOM_PROFIL_LOCATION: &str = nom_de_variante!(Location);

/// Profils de machine. Les variantes sont des identifiants typés ; leur nom en minuscules est la clé du profil dans
/// `kvu.toml`, où sont écrits les motifs de reconnaissance, les chemins et les seuils. La détection (hôte, CPU, GPU)
/// est dans `machine::detecter` ; `Location` est le profil d'une machine louée (conteneur, sans câble ni ssh).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Machine {
    /// Machine principale (APU AMD, GPU HIP).
    Acemagic,
    /// Machine secondaire (GPU NVIDIA, cœurs réservés).
    Gamer,
    /// Toute autre machine : politique prudente, backend détecté.
    Inconnue,
    /// Machine louée : aucun nom d'hôte, aucun câble, verrou local, store sous `$HOME`.
    Location,
}

impl Machine {
    /// Tous les profils, dans l'ordre de la reconnaissance (les profils sans motif ne sont jamais reconnus).
    pub const TOUS: [Machine; 4] = [Machine::Acemagic, Machine::Gamer, Machine::Inconnue, Machine::Location];

    /// Clé du profil dans `kvu.toml` (aussi le nom affiché par `doctor`).
    pub fn nom(self) -> &'static str {
        match self {
            Machine::Acemagic => NOM_PROFIL_ACEMAGIC,
            Machine::Gamer => NOM_PROFIL_GAMER,
            Machine::Inconnue => NOM_PROFIL_INCONNUE,
            Machine::Location => NOM_PROFIL_LOCATION,
        }
    }

    /// Profil générique : sans motif de reconnaissance, il peut se superposer à n'importe quel matériel (le verrou GPU,
    /// l'affinité et les planchers restent alors ceux du matériel réel).
    pub fn est_generique(self) -> bool {
        matches!(self, Machine::Inconnue | Machine::Location)
    }

    /// Les noms de profils séparés par `|` (messages d'aide et d'erreur).
    pub fn noms_possibles() -> String {
        Machine::TOUS.iter().map(|m| m.nom()).collect::<Vec<_>>().join("|")
    }
}

impl FromStr for Machine {
    type Err = Erreur;
    fn from_str(s: &str) -> Resultat<Self> {
        let voulu = s.trim().to_ascii_lowercase();
        Machine::TOUS.iter().copied().find(|m| m.nom() == voulu).ok_or_else(|| {
            Erreur::Config(format!(
                "machine inconnue « {voulu} » ({})",
                Machine::noms_possibles()
            ))
        })
    }
}

/// Précision du générateur.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// Chemin de RÉFÉRENCE : FP64 partout, expressions identiques à su2_hip.cpp / su2_cuda.cu.
    F64,
    /// Chemin RAPIDE : stockage et arithmétique FP32, réunitarisation à chaque mise à jour,
    /// observables accumulées en FP64. Même loi que F64 (gates LOI-*), pas les mêmes bits.
    F32,
}

impl Precision {
    pub fn nom(self) -> &'static str {
        match self {
            Precision::F64 => "f64",
            Precision::F32 => "f32",
        }
    }
}

impl FromStr for Precision {
    type Err = Erreur;
    fn from_str(s: &str) -> Resultat<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f64" | "fp64" | "ref" => Ok(Precision::F64),
            "f32" | "fp32" | "rapide" => Ok(Precision::F32),
            autre => Err(Erreur::Config(format!("précision inconnue « {autre} » (f64|f32)"))),
        }
    }
}

/// Départ d'une chaîne.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Depart {
    /// Liens Haar i.i.d. (Philox, compteur (lien, 0xFFFFFFFF, 0|1, ETIQUETTE_INIT)).
    Chaud,
    /// Liens = identité.
    Froid,
}

/// Graine 64 bits ; la clé Philox est (graine basse, graine haute), comme su2_hip.cpp.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Graine(pub u64);

impl Graine {
    /// Clé Philox4x32-10 (k0, k1).
    pub fn cle(self) -> (u32, u32) {
        ((self.0 & 0xFFFF_FFFF) as u32, (self.0 >> 32) as u32)
    }
}

// ------------------------------------------------------------------------------------------
// Étiquettes du compteur Philox (4ᵉ mot du compteur). Les trois premières viennent de
// su2_hip.cpp / su2_bord.cu et NE DOIVENT PAS changer (bit-identité).
// ------------------------------------------------------------------------------------------
pub const ETIQUETTE_INIT: u32 = 0x494E_4954; // "INIT"
pub const ETIQUETTE_HB: u32 = 0x4842_4154; // "HBAT"
pub const ETIQUETTE_BORD: u32 = 0x424F_5244; // "BORD" (su2_bord.cu)
pub const ETIQUETTE_JAUGE: u32 = 0x4A41_5547; // "JAUG" (su2_bord.cu)
pub const ETIQUETTE_COND: u32 = 0x434F_4E44; // "COND" (bain conditionnel, mode rapide)

// ------------------------------------------------------------------------------------------
// Champ de jauge côté hôte
// ------------------------------------------------------------------------------------------

/// Champ SU(2) L⁴ périodique sur l'hôte, en quaternions FP64.
///
/// Convention (celle de su2_hip.cpp, de crate::heritage::su2_hb et de noyau2/cb.py) :
/// - U = a0 + i(a1 σ1 + a2 σ2 + a3 σ3), stocké (x=a0, y=a1, z=a2, w=a3) ;
///   matrice [[a0 + i a3, a2 + i a1], [-a2 + i a1, a0 - i a3]] ;
/// - lien li = mu*V + s, s = ((x0*L + x1)*L + x2)*L + x3, lien de x vers x + e_mu ;
/// - direction « temporelle » des Polyakov et des boucles R×T : 3 par défaut ;
///   la profondeur du bord gelé (R2-3) est x0.
#[derive(Clone, Debug, PartialEq)]
pub struct ChampHote {
    pub l: usize,
    /// 16·V réels : 4 directions × V sites × 4 composantes.
    pub q: Vec<f64>,
}

impl ChampHote {
    /// Champ froid (toutes les matrices = identité). L pair ≥ 2.
    pub fn froid(l: usize) -> Resultat<Self> {
        if l < 2 || !l.is_multiple_of(2) {
            return Err(Erreur::Config(format!("L = {l} : L pair ≥ 2 requis")));
        }
        let v = l * l * l * l;
        let mut q = vec![0.0; 16 * v];
        for lien in q.chunks_exact_mut(4) {
            lien[0] = 1.0;
        }
        Ok(ChampHote { l, q })
    }

    /// Nombre de sites L⁴.
    pub fn v(&self) -> usize {
        self.l * self.l * self.l * self.l
    }

    /// Quaternion du lien `li`.
    pub fn lien(&self, li: usize) -> [f64; 4] {
        let o = 4 * li;
        [self.q[o], self.q[o + 1], self.q[o + 2], self.q[o + 3]]
    }

    /// max | |q| − 1 | sur tous les liens.
    pub fn defaut_unitarite(&self) -> f64 {
        self.q
            .chunks_exact(4)
            .map(|c| ((c[0] * c[0] + c[1] * c[1] + c[2] * c[2] + c[3] * c[3]).sqrt() - 1.0).abs())
            .fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graine_cle_comme_su2_hip() {
        let g = Graine(0x0000_0006_4C8B_3F2A);
        assert_eq!(g.cle(), (0x4C8B_3F2A, 0x0000_0006));
    }

    #[test]
    fn analyse_des_noms() {
        assert_eq!("HIP".parse::<BackendKind>().unwrap(), BackendKind::Hip);
        assert_eq!("fp32".parse::<Precision>().unwrap(), Precision::F32);
        assert!("vulkan".parse::<BackendKind>().is_err());
        assert_eq!("gamer".parse::<Machine>().unwrap(), Machine::Gamer);
    }

    #[test]
    fn champ_froid_unitaire() {
        let c = ChampHote::froid(4).unwrap();
        assert_eq!(c.q.len(), 16 * 256);
        assert_eq!(c.defaut_unitarite(), 0.0);
        assert!(ChampHote::froid(3).is_err());
    }

    #[test]
    fn codes_de_sortie_stables() {
        assert_eq!(Erreur::Verrou("x".into()).code_sortie(), 6);
        assert_eq!(
            Erreur::SsdPlein { chemin: "/".into(), libre_go: 1.0, besoin_go: 1.0, seuil_go: 100.0 }.code_sortie(),
            7
        );
    }
}
