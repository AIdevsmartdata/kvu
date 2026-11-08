// SPDX-License-Identifier: Apache-2.0
//! Mesures GPU des boîtes de Dirichlet ℓ⁴ (module de noyaux « boite », source unique HIP/CUDA). Propriétaire : WP-DIR.
//!
//! Le module `boite` est compilé à part du module `su2_ref64` : socle `su2_commun.h` + `su2_boite.cu`, options `ref-k1`
//! (aucune contraction multiplication-addition), de sorte que les sommes de la plaquette centrale, de la boucle 2×2 centrale
//! et du profil par couche égalent à l'octet la référence hôte [`super::boite::mesurer_hote`] (mêmes ordres de sommation).
//! Le fichier `su2_boite.cu` est embarqué ici (`include_str!`) et non dans le registre gelé `crate::noyaux::TOUS`
//! (dont la longueur est figée par des tests existants) ; il est couvert par la recette `kvu-source/1` (dossier `noyaux/`)
//! et sa source assemblée figure dans les `modules_compiles` des manifestes.
//!
//! Disposition des historiques (par lot de `nch` chaînes, capacité `capacite` mesures) :
//! `centre[(c·capacite + idx)·2 + {0 : P_c, 1 : W2}]` et `profil[(c·capacite + idx)·ncl + d]` = SOMME de ½Tr de la classe d.

use super::boite::{classes_profil, ClassesProfil, GeometrieBoite, ObsBoite};
use super::champ::{ChampGpu, TamponChamp};
use crate::commun::{BackendKind, Erreur, Resultat};
use crate::gpu::{assembler, Appareil, Arg, Fonction, Lancement, Module, OptionsCompil, ProvenanceCompil, SourceAssemblee, Tampon};
use crate::noyaux::{Fichier, SU2_COMMUN};

/// Source GPU des mesures de boîte (voir l'en-tête du fichier pour les signatures et les ordres de sommation).
pub static SU2_BOITE: Fichier = Fichier { nom: "su2_boite.cu", contenu: include_str!("../../noyaux/su2_boite.cu") };

/// Taille de bloc de `k_boite_profil` (arbre fixe à 256 fils).
const BLOC_PROFIL: u32 = 256;
/// Taille de bloc de `k_boite_centre` (une plaquette ou une boucle par fil).
const BLOC_CENTRE: u32 = 32;

/// Source assemblée du module « boite » (testable sans GPU).
pub fn source_boite(backend: BackendKind) -> SourceAssemblee {
    assembler(backend, "boite", &[&SU2_COMMUN, &SU2_BOITE], &[])
}

/// Mesures de boîte sur un champ résident F64 (un lot de `nch` chaînes de même ℓ).
pub struct MesureBoiteGpu {
    app: Appareil,
    module: Module,
    f_centre: Fonction,
    f_profil: Fonction,
    g: GeometrieBoite,
    classes: ClassesProfil,
    entrees: Tampon<u32>,
    debuts: Tampon<u32>,
    nch: usize,
    capacite: usize,
    centre: Tampon<f64>,
    profil: Tampon<f64>,
}

impl MesureBoiteGpu {
    /// Compile (ou relit du cache) le module et alloue les tampons : `nch` chaînes, `capacite` mesures conservées avant lecture.
    pub fn nouveau(app: &Appareil, g: &GeometrieBoite, nch: usize, capacite: usize) -> Resultat<MesureBoiteGpu> {
        if nch == 0 || capacite == 0 {
            return Err(Erreur::Config("mesures de boîte : au moins une chaîne et une case d'historique requises".into()));
        }
        let src = source_boite(app.backend());
        let module = app.compiler(&src, &OptionsCompil::reference_k1())?;
        let classes = classes_profil(g);
        let mut entrees = app.alloc::<u32>(classes.entrees.len().max(1))?;
        app.ecrire(&mut entrees, 0, &classes.entrees)?;
        let mut debuts = app.alloc::<u32>(classes.debuts.len())?;
        app.ecrire(&mut debuts, 0, &classes.debuts)?;
        let ncl = classes.nclasses;
        let m = MesureBoiteGpu {
            f_centre: app.fonction(&module, "k_boite_centre")?,
            f_profil: app.fonction(&module, "k_boite_profil")?,
            centre: app.alloc::<f64>(nch * capacite * 2)?,
            profil: app.alloc::<f64>(nch * capacite * ncl)?,
            app: app.clone(),
            module,
            g: *g,
            classes,
            entrees,
            debuts,
            nch,
            capacite,
        };
        app.synchroniser()?;
        Ok(m)
    }

    /// Provenance du module compilé (source assemblée, options, code objet), pour le manifeste.
    pub fn provenance(&self) -> ProvenanceCompil {
        self.module.provenance().clone()
    }

    pub fn classes(&self) -> &ClassesProfil {
        &self.classes
    }

    pub fn capacite(&self) -> usize {
        self.capacite
    }

    /// Mesure TOUTES les chaînes du champ (asynchrone, sans synchronisation) et range le résultat à la case `idx`.
    /// `avec_profil = false` saute le profil (la case de profil garde son ancienne valeur).
    pub fn mesurer(&self, champ: &ChampGpu, idx: usize, avec_profil: bool) -> Resultat<()> {
        if idx >= self.capacite {
            return Err(Erreur::Config(format!("historique de mesures plein : idx {idx} ≥ capacité {}", self.capacite)));
        }
        if champ.l != self.g.lc || champ.nchaines != self.nch {
            return Err(Erreur::Config(format!(
                "mesures de boîte : champ (L = {}, {} chaînes) incompatible avec la boîte (L_c = {}, {} chaînes)",
                champ.l, champ.nchaines, self.g.lc, self.nch
            )));
        }
        let u = match &champ.tampon {
            TamponChamp::F64(t) => t.arg(),
            TamponChamp::F32(_) => return Err(Erreur::Config("mesures de boîte : champ F64 requis".into())),
        };
        let ncl = self.classes.nclasses;
        let lc = self.g.lc as i32;
        self.app.lancer(
            &self.f_centre,
            &Lancement { grille: [1, self.nch as u32, 1], bloc: [BLOC_CENTRE, 1, 1], shmem: 0 },
            &[u, Arg::I32(lc), self.centre.arg(), Arg::I64((2 * self.capacite) as i64), Arg::I64(idx as i64)],
        )?;
        if avec_profil {
            self.app.lancer(
                &self.f_profil,
                &Lancement { grille: [ncl as u32, self.nch as u32, 1], bloc: [BLOC_PROFIL, 1, 1], shmem: 0 },
                &[
                    u,
                    Arg::I32(lc),
                    self.entrees.arg(),
                    self.debuts.arg(),
                    self.profil.arg(),
                    Arg::I64((ncl * self.capacite) as i64),
                    Arg::I64(idx as i64),
                    Arg::I32(ncl as i32),
                ],
            )?;
        }
        Ok(())
    }

    /// Lit (en synchronisant) les deux historiques : `(centre, profil)`, tailles `nch·capacite·2` et `nch·capacite·ncl`
    /// (sommes pour le profil : l'hôte divise par `classes().comptes`).
    pub fn lire(&self) -> Resultat<(Vec<f64>, Vec<f64>)> {
        self.app.synchroniser()?;
        let mut c = vec![0f64; self.centre.len()];
        let mut p = vec![0f64; self.profil.len()];
        self.app.lire(&self.centre, 0, &mut c)?;
        self.app.lire(&self.profil, 0, &mut p)?;
        Ok((c, p))
    }

    /// Mesure et lit les observables de la chaîne `chaine` du champ (case 0 ; pour les tests et `su2 boite-obs`).
    pub fn observables(&self, champ: &ChampGpu, chaine: usize) -> Resultat<ObsBoite> {
        if chaine >= self.nch {
            return Err(Erreur::Config(format!("chaîne {chaine} hors du lot ({} chaînes)", self.nch)));
        }
        self.mesurer(champ, 0, true)?;
        let (c, p) = self.lire()?;
        Ok(self.extraire(&c, &p, chaine, 0))
    }

    /// Observables (profil en moyennes) de la chaîne `chaine`, case `idx`, à partir des historiques lus.
    pub fn extraire(&self, centre: &[f64], profil: &[f64], chaine: usize, idx: usize) -> ObsBoite {
        let ncl = self.classes.nclasses;
        let o2 = (chaine * self.capacite + idx) * 2;
        let op = (chaine * self.capacite + idx) * ncl;
        ObsBoite {
            pc: centre[o2],
            w2: if self.g.ell >= 4 { centre[o2 + 1] } else { f64::NAN },
            profil: (0..ncl).map(|d| profil[op + d] / self.classes.comptes[d] as f64).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assemblage_du_module_boite_pour_les_deux_backends() {
        for b in [BackendKind::Hip, BackendKind::Cuda] {
            let s = source_boite(b);
            let noms: Vec<&str> = s.parties.iter().map(|(n, _)| n.as_str()).collect();
            // prélude du backend, puis le socle commun, puis les mesures de boîte (dans cet ordre)
            let prelude = if b == BackendKind::Hip { "prelude_hip.h" } else { "prelude_cuda.h" };
            assert_eq!(noms, vec![prelude, "su2_commun.h", "su2_boite.cu"], "{noms:?}");
            for k in ["k_boite_centre", "k_boite_profil"] {
                assert!(s.texte.contains(&format!("void {k}(")), "noyau {k} absent ({})", b.nom());
            }
            assert_eq!(source_boite(b).sha256, s.sha256, "assemblage déterministe");
            assert!(!s.texte.contains("__constant__"), "le runtime n'a pas de mémoire constante");
        }
    }

    #[test]
    fn le_noyau_declare_les_ordres_de_sommation_contractuels() {
        let t = SU2_BOITE.contenu;
        for motif in ["s / 24.0", "w / 6.0", "i += 256", "for (int w = 128; w > 0; w >>= 1)"] {
            assert!(t.contains(motif), "motif absent de su2_boite.cu : {motif}");
        }
    }
}
