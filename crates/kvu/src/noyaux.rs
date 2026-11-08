//! Registre des sources GPU embarquées (source UNIQUE pour HIP et CUDA).
//!
//! GELÉ (architecte) : la LISTE des fichiers est gelée ; le CONTENU de chaque fichier
//! appartient à la tâche indiquée. Le manifeste de provenance hache chacun de ces fichiers
//! (`empreintes`) ET chaque source effectivement assemblée et compilée (`gpu::SourceAssemblee`).
//!
//! Assemblage (tâche T2, `gpu::assembler`) : prélude du backend, puis `#define` éventuels,
//! puis les fichiers dans l'ordre demandé. Aucune directive `#include` entre fichiers du
//! registre : l'ordre d'assemblage tient lieu d'include (hiprtc et NVRTC voient le même texte).

/// Un fichier source GPU embarqué.
#[derive(Debug)]
pub struct Fichier {
    pub nom: &'static str,
    pub contenu: &'static str,
}

macro_rules! fichier {
    ($ident:ident, $nom:literal) => {
        pub static $ident: Fichier = Fichier { nom: $nom, contenu: include_str!(concat!("../noyaux/", $nom)) };
    };
}

// Préludes et essai du runtime [T2]
fichier!(PRELUDE_HIP, "prelude_hip.h");
fichier!(PRELUDE_CUDA, "prelude_cuda.h");
fichier!(TEST_RUNTIME, "test_runtime.cu");
// Générateur SU(2) [T3] et chemin FP32 [T4]
fichier!(SU2_COMMUN, "su2_commun.h");
fichier!(SU2_REF64, "su2_ref64.cu");
fichier!(SU2_BORD, "su2_bord.cu");
fichier!(SU2_F32, "su2_f32.cu");
// Bain conditionnel [T5] et mesure de fibre [T6]
fichier!(COND, "cond.cu");
fichier!(FIBRE, "fibre.cu");
// Observables [T7]
fichier!(OBS_WILSON, "obs_wilson.cu");
fichier!(OBS_FLOT, "obs_flot.cu");
fichier!(RG_BLOC, "rg_bloc.cu");

/// Tous les fichiers embarqués, dans un ordre stable (celui du manifeste).
pub static TOUS: &[&Fichier] = &[
    &PRELUDE_HIP,
    &PRELUDE_CUDA,
    &TEST_RUNTIME,
    &SU2_COMMUN,
    &SU2_REF64,
    &SU2_BORD,
    &SU2_F32,
    &COND,
    &FIBRE,
    &OBS_WILSON,
    &OBS_FLOT,
    &RG_BLOC,
];

/// (nom, sha256 du contenu) de chaque fichier embarqué, pour le manifeste de provenance.
pub fn empreintes() -> Vec<(&'static str, String)> {
    TOUS.iter()
        .map(|f| (f.nom, crate::heritage::manifest::sha256_hex(f.contenu.as_bytes())))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registre_complet_et_noms_uniques() {
        let mut noms: Vec<&str> = TOUS.iter().map(|f| f.nom).collect();
        let n = noms.len();
        noms.sort_unstable();
        noms.dedup();
        assert_eq!(noms.len(), n, "noms de noyaux dupliqués");
        assert!(TOUS.iter().all(|f| !f.contenu.is_empty()));
        assert_eq!(empreintes().len(), n);
        assert!(empreintes().iter().all(|(_, h)| h.len() == 64));
    }
}
