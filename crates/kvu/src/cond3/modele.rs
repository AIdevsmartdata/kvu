//! Interface de groupe du bain conditionnel de facteur 3 (SPEC_T3 §3.1 « interface de groupe (obligatoire) »).
//!
//! Le même code de géométrie ([`super::geom`]), d'ordonnancement ([`Modele::pas`]), de mesure et d'estimation
//! ([`super::mesure`]) sert à SU(2) ([`super::su2::FibreSu2`]) et au U(1) gaussien non compact
//! ([`super::u1::FibreU1`]) ; seules les mises à jour locales, les agrafes et les scores sont propres au groupe.

use super::geom::Geom;
use std::sync::Arc;

/// Diagnostics cumulés d'un échantillonneur.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StatsEchantillonneur {
    /// Mises à jour de bain thermique effectuées (CPU : tous les balayages ; GPU : balayages de CONTRÔLE seulement, échantillon).
    pub majs: u64,
    /// Essais cumulés des tirages de rejet (Kennedy–Pendleton) ; 0 en U(1) (même périmètre que `majs`).
    pub essais: u64,
    /// Tirages ayant atteint le plafond d'essais (attendu 0) ; cumulé sur TOUS les balayages, CPU et GPU.
    pub plafond: u64,
}

/// Mesure supplémentaire d'oracle, évaluée à chaque itération mesurée (séries temporelles `serie_canaux`).
#[derive(Clone, Debug, PartialEq)]
pub enum Canal {
    /// Identité de Stein sur un lien LIBRE `li` (multiplication à gauche par exp(itσ_a)) : [Y(O_p), Y(S), Y²(S)] pour la plaquette `pi`
    /// (Y²(S) pour l'identité de Fisher E[Y²(S)] = Var(Y(S))).
    SteinLibre { li: u32, a: u8, pi: u32 },
    /// Identité de Stein pour le mouvement vertical de la ligne `ci` : (U₁ → U₁ e^{itσ_a}, U₂ → e^{−itσ_a} U₂) : [Y(O_p), Y(S), Y²(S)].
    SteinLigne { ci: u32, a: u8, pi: u32 },
    /// ΔS_t du flot du relèvement 1 (U₁ → e^{itσ_a} U₁) sur la ligne `ci` : [ΔS_t] (repondération exacte, SPEC §2.8).
    DeltaS { ci: u32, a: u8, t: f64 },
    /// Moments des liens de ligne : [⟨½Tr U_j⟩ (j = 1, 2, 3), ⟨(½Tr U_j)²⟩ (j = 1, 2, 3)] moyennés sur toutes les lignes (oracle O-5).
    MomentsLignes,
    /// Moyennes de O sur TOUTES les plaquettes par classe t (0..=d−2) puis par nombre de liens de ligne (0, 1, 2).
    GroupesPlaquettes,
    /// Observable O_p d'une plaquette : [O_p].
    Plaq { pi: u32 },
}

impl Canal {
    /// Nombre de valeurs produites par évaluation (d = dimension du réseau).
    pub fn largeur(&self, d: usize) -> usize {
        match self {
            Canal::SteinLibre { .. } | Canal::SteinLigne { .. } => 3,
            Canal::DeltaS { .. } | Canal::Plaq { .. } => 1,
            Canal::MomentsLignes => 6,
            Canal::GroupesPlaquettes => (d - 1) + 3,
        }
    }
}

/// Un échantillonneur de la fibre dure : champ, V figé, β, flux d'aléa.
pub trait Modele: Send + Sync {
    fn geom(&self) -> &Arc<Geom>;
    fn beta(&self) -> f64;
    /// Nombre de composantes d'un score de relèvement (3 en SU(2), 1 en U(1)).
    fn ncomp(&self) -> usize;
    /// Nom du groupe (« su2 » ou « u1 »), pour les sorties.
    fn groupe(&self) -> &'static str;

    /// Un pas CONTRAINT : 1 balayage de bain thermique puis `n_or` balayages de surrelaxation (ordre du SPEC §3.1 : pour mu = 0..d,
    /// (a) libres de parité paire, (b) impaire, (c) tous les U₁ des lignes, (d) tous les U₂ ; U₃ asservi).
    fn pas(&mut self, sweep: u32, n_or: usize);
    /// Même loi, autre ordonnancement : mise à jour SÉQUENTIELLE lien par lien, en place (oracle O-2b).
    fn pas_sequentiel(&mut self, sweep: u32, n_or: usize);
    /// Un pas SANS contrainte (tous les liens libres, damier) : thermalisation de Wilson / mutant MUT-SANS-CONTRAINTE.
    fn pas_libre(&mut self, sweep: u32, n_or: usize);

    /// Fige V = DEC(U) à partir du champ courant (départ exact de la fibre).
    fn figer_v(&mut self);

    /// Observables O_p des plaquettes `pis` (½ Tr U_p en SU(2), F_p en U(1)).
    fn observables(&self, pis: &[u32], out: &mut [f64]);
    /// Scores des sondes `cis` : `out[s·nc + k]` avec nc = ncomp (moyen des trois relèvements) ou 3·ncomp (relèvements séparés
    /// σ¹, σ², σ³ concaténés) si `separes`.
    fn scores(&self, cis: &[u32], separes: bool, out: &mut [f64]);

    /// Terme X̄(O_p) (moyenne des trois relèvements de X(O_p), SPEC §2.2) pour des paires (plaquette `pi`, sonde `ci`) :
    /// `out[q·nc + c]`, nc = ncomp (ou 3·ncomp si `separes`) ; nul si la plaquette ne contient aucun lien de la ligne.
    fn terme_x(&self, paires: &[(u32, u32)], separes: bool, out: &mut [f64]);

    /// Évalue un canal d'oracle ; faux (et `out` inchangé) si le modèle ne le gère pas.
    fn canal(&self, _c: &Canal, _out: &mut [f64]) -> bool {
        false
    }

    /// max |DEC(U) − V| sur toutes les lignes.
    fn residu_contrainte(&self) -> f64;
    /// max |identité de Ward grossière| (SPEC §2.2) sur tous les sites grossiers.
    fn residu_ward(&self) -> f64;
    /// max ||q| − 1| (SU(2)) ; 0 en U(1).
    fn defaut_unitarite(&self) -> f64;
    fn stats(&self) -> StatsEchantillonneur;
    /// sha256 (hex) de l'état du champ (octets IEEE des composantes) : rejeu au bit (G-REJEU).
    fn empreinte(&self) -> String;
    /// Copie du champ en f64 (composantes aplaties) : tests et oracles.
    fn champ_plat(&self) -> Vec<f64>;
    /// V figé aplati.
    fn v_fige_plat(&self) -> Vec<f64>;
}

/// Type de pas choisi par le pilote d'un oracle ou d'une production.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ordonnancement {
    /// Vectorisé (SPEC §3.1) : sous-étapes simultanées.
    Simultane,
    /// Séquentiel lien par lien (oracle O-2b).
    Sequentiel,
    /// Sans contrainte (thermalisation de Wilson).
    Libre,
}

pub fn avancer<M: Modele + ?Sized>(m: &mut M, ord: Ordonnancement, sweep: u32, n_or: usize) {
    match ord {
        Ordonnancement::Simultane => m.pas(sweep, n_or),
        Ordonnancement::Sequentiel => m.pas_sequentiel(sweep, n_or),
        Ordonnancement::Libre => m.pas_libre(sweep, n_or),
    }
}
