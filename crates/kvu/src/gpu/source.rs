//! Assemblage des sources (prélude + defines + fichiers du registre) et options de compilation.
//! Propriétaire : T2.
//!
//! Tout ce module est PUR (aucun GPU, aucune E/S) : l'assemblage et les arguments de compilation sont
//! déterministes et testés par des valeurs d'or (`tests/gpu_source.rs`).

use crate::commun::{BackendKind, Erreur, Resultat};
use crate::heritage::manifest::sha256_hex;
use crate::noyaux::{Fichier, PRELUDE_CUDA, PRELUDE_HIP};

/// Contraction des multiplications-additions (HIP : -ffp-contract ; CUDA : --fmad).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FpContract {
    /// Aucune contraction (gate K1 : su2_hip.cpp compilé -ffp-contract=off ; CUDA --fmad=false).
    Off,
    On,
    Fast,
    /// Défaut de hipcc pour le HIP (binaire de production su2_hip).
    FastHonorPragmas,
}

impl FpContract {
    /// Valeur de l'option `-ffp-contract=` de clang/hiprtc.
    pub fn nom_hip(self) -> &'static str {
        match self {
            FpContract::Off => "off",
            FpContract::On => "on",
            FpContract::Fast => "fast",
            FpContract::FastHonorPragmas => "fast-honor-pragmas",
        }
    }
}

/// Options de compilation, hachées dans le manifeste et dans la clé du cache.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OptionsCompil {
    /// HIP : -ffp-contract=… ; CUDA : Off ⇒ --fmad=false, sinon --fmad=true.
    pub fp_contract: FpContract,
    /// Niveau d'optimisation (-O3 par défaut).
    pub optimisation: u8,
    /// Arguments supplémentaires, dans l'ordre (ex. « -DK1_MUTANT_METRO=1 » pour un mutant).
    pub extra: Vec<String>,
}

impl OptionsCompil {
    /// Référence K1 : HIP -O3 -ffp-contract=off ; CUDA --fmad=false. Bit-identique à su2_gen (K1).
    pub fn reference_k1() -> Self {
        OptionsCompil {
            fp_contract: FpContract::Off,
            optimisation: 3,
            extra: Vec::new(),
        }
    }
    /// Compatibilité des binaires de production : HIP fast-honor-pragmas (hipcc par défaut,
    /// su2_hip) ; CUDA --fmad=true (nvcc par défaut, su2_cuda / su2_bord).
    pub fn compat_production(backend: BackendKind) -> Self {
        let fp = match backend {
            BackendKind::Hip => FpContract::FastHonorPragmas,
            _ => FpContract::Fast,
        };
        OptionsCompil {
            fp_contract: fp,
            optimisation: 3,
            extra: Vec::new(),
        }
    }
    /// Chemin rapide FP32 : contraction autorisée partout.
    pub fn rapide() -> Self {
        OptionsCompil {
            fp_contract: FpContract::Fast,
            optimisation: 3,
            extra: Vec::new(),
        }
    }

    /// Ajoute un argument supplémentaire (ordre conservé).
    pub fn avec_extra(mut self, arg: impl Into<String>) -> Self {
        self.extra.push(arg.into());
        self
    }

    /// Arguments effectifs du compilateur pour `backend` et `arch` (testable sans GPU).
    ///
    /// - HIP : `--offload-arch=<arch> -O<n> -ffp-contract=<…> -std=c++17` (hiprtc n'infère pas l'architecture) ;
    /// - CUDA : `-arch=<sm_XY> --fmad=<true|false> -std=c++17`. Le préfixe `sm_` fait produire un CUBIN
    ///   (et non du PTX). NVRTC n'a pas d'option -O : ptxas optimise en O3 par défaut, et un niveau
    ///   différent est transmis par `--ptxas-options=-O<n>` ;
    /// - Cpu : aucun argument.
    ///
    /// Les arguments `extra` suivent, dans l'ordre.
    pub fn arguments(&self, backend: BackendKind, arch: &str) -> Vec<String> {
        let mut a = Vec::with_capacity(4 + self.extra.len());
        match backend {
            BackendKind::Hip => {
                a.push(format!("--offload-arch={arch}"));
                a.push(format!("-O{}", self.optimisation));
                a.push(format!("-ffp-contract={}", self.fp_contract.nom_hip()));
                a.push("-std=c++17".to_string());
            }
            BackendKind::Cuda => {
                a.push(format!("-arch={arch}"));
                a.push(format!("--fmad={}", self.fp_contract != FpContract::Off));
                a.push("-std=c++17".to_string());
                if self.optimisation != 3 {
                    a.push(format!("--ptxas-options=-O{}", self.optimisation));
                }
            }
            BackendKind::Cpu => return Vec::new(),
        }
        a.extend(self.extra.iter().cloned());
        a
    }
}

/// Source prête à compiler, avec sa provenance.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SourceAssemblee {
    /// Nom logique (« su2_ref64 », « cond »…), sert au cache et aux journaux.
    pub nom: String,
    pub backend: BackendKind,
    pub texte: String,
    /// sha256 du texte assemblé.
    pub sha256: String,
    /// (fichier, sha256) de chaque partie, dans l'ordre d'assemblage (prélude compris).
    pub parties: Vec<(String, String)>,
}

/// Prélude du backend (None pour le CPU).
fn prelude(backend: BackendKind) -> Option<&'static Fichier> {
    match backend {
        BackendKind::Hip => Some(&PRELUDE_HIP),
        BackendKind::Cuda => Some(&PRELUDE_CUDA),
        BackendKind::Cpu => None,
    }
}

/// Ligne de séparation placée avant chaque fichier (prélude compris).
fn separateur(nom: &str) -> String {
    format!("// ---- {nom} ----\n")
}

/// Un nom de macro valide : [A-Za-z_][A-Za-z0-9_]*.
fn nom_de_macro_valide(nom: &str) -> bool {
    let mut c = nom.chars();
    matches!(c.next(), Some(p) if p.is_ascii_alphabetic() || p == '_')
        && c.all(|x| x.is_ascii_alphanumeric() || x == '_')
}

/// Variante faillible de [`assembler`] : refuse le backend CPU, un nom de macro invalide
/// ou une valeur de macro multiligne.
pub fn essayer_assembler(
    backend: BackendKind,
    nom: &str,
    fichiers: &[&Fichier],
    defines: &[(&str, &str)],
) -> Resultat<SourceAssemblee> {
    let pre = prelude(backend)
        .ok_or_else(|| Erreur::Config(format!("assemblage « {nom} » : le backend cpu n'a pas de source GPU")))?;
    for (k, v) in defines {
        if !nom_de_macro_valide(k) {
            return Err(Erreur::Config(format!(
                "assemblage « {nom} » : nom de macro invalide « {k} »"
            )));
        }
        if v.contains('\n') || v.contains('\r') {
            return Err(Erreur::Config(format!(
                "assemblage « {nom} » : valeur multiligne pour la macro {k}"
            )));
        }
    }

    let mut texte = String::new();
    let mut parties = Vec::with_capacity(fichiers.len() + 1);
    let mut ajouter = |texte: &mut String, f: &Fichier| {
        texte.push_str(&separateur(f.nom));
        texte.push_str(f.contenu);
        if !f.contenu.ends_with('\n') {
            texte.push('\n');
        }
        parties.push((f.nom.to_string(), sha256_hex(f.contenu.as_bytes())));
    };
    ajouter(&mut texte, pre);
    for (k, v) in defines {
        if v.is_empty() {
            texte.push_str(&format!("#define {k}\n"));
        } else {
            texte.push_str(&format!("#define {k} {v}\n"));
        }
    }
    for f in fichiers {
        ajouter(&mut texte, f);
    }
    let sha256 = sha256_hex(texte.as_bytes());
    Ok(SourceAssemblee {
        nom: nom.to_string(),
        backend,
        texte,
        sha256,
        parties,
    })
}

/// Prélude du backend, puis une ligne `#define NOM VALEUR` par define, puis les fichiers.
/// Chaque fichier (prélude compris) est précédé d'une ligne `// ---- <nom> ----`.
/// Le texte est déterministe (même entrée ⇒ mêmes octets ⇒ même sha256).
///
/// # Panics
/// Si `backend` est le CPU ou si un define est invalide : ce sont des erreurs de programmation
/// (voir [`essayer_assembler`] pour la variante qui renvoie une erreur).
pub fn assembler(backend: BackendKind, nom: &str, fichiers: &[&Fichier], defines: &[(&str, &str)]) -> SourceAssemblee {
    match essayer_assembler(backend, nom, fichiers, defines) {
        Ok(s) => s,
        Err(e) => panic!("assembler : {e}"),
    }
}

/// Provenance d'un module compilé, pour le manifeste (une entrée par source chargée).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProvenanceCompil {
    pub nom: String,
    pub backend: BackendKind,
    pub arch: String,
    pub sha256_source: String,
    pub parties: Vec<(String, String)>,
    pub options: OptionsCompil,
    pub arguments: Vec<String>,
    pub sha256_code: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    static F1: Fichier = Fichier {
        nom: "a.cu",
        contenu: "int a;\n",
    };
    static F2: Fichier = Fichier {
        nom: "b.cu",
        contenu: "int b;",
    }; // sans retour final

    #[test]
    fn nom_de_macro() {
        assert!(nom_de_macro_valide("K1_MUTANT_METRO"));
        assert!(nom_de_macro_valide("_x9"));
        assert!(!nom_de_macro_valide(""));
        assert!(!nom_de_macro_valide("9x"));
        assert!(!nom_de_macro_valide("a b"));
        assert!(!nom_de_macro_valide("a-b"));
    }

    #[test]
    fn retour_final_ajoute_sans_doublon() {
        let s = assembler(BackendKind::Cuda, "t", &[&F1, &F2], &[]);
        assert!(s.texte.ends_with("// ---- b.cu ----\nint b;\n"));
        assert!(s.texte.contains("// ---- a.cu ----\nint a;\n// ---- b.cu ----\n"));
    }

    #[test]
    fn define_vide_et_valide() {
        let s = assembler(BackendKind::Hip, "t", &[&F1], &[("A", ""), ("B", "3")]);
        assert!(s.texte.contains("#define A\n#define B 3\n// ---- a.cu ----"));
    }

    #[test]
    fn refus_cpu_et_define_invalide() {
        assert!(essayer_assembler(BackendKind::Cpu, "t", &[&F1], &[]).is_err());
        assert!(essayer_assembler(BackendKind::Hip, "t", &[&F1], &[("a b", "1")]).is_err());
        assert!(essayer_assembler(BackendKind::Hip, "t", &[&F1], &[("A", "1\n2")]).is_err());
    }
}
