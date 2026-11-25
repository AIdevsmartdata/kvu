//! DOMAINE ADMIS des options numériques du module tube (révision 3, relecture c16b) : chaque option reçoit un intervalle
//! explicite, confronté dans les fonctions `analyser_*` de `cli.rs` (qui lisent toute la ligne de commande), AVANT les
//! grosses allocations (champs, piles, chaînes) et AVANT toute écriture. Garanti : hors du domaine, refus d'usage (code 2),
//! rien d'écrit et aucune allocation de champ ; NON garanti : les petites listes issues des arguments (par exemple la liste des
//! β) sont construites avant d'être comptées et refusées. La table [`DOMAINE`] porte les intervalles que reprennent l'aide
//! (`kvu su2 tube --help`), la note `docs/v0.2/notes/tube.md` et le test paramétré `tests/su2_tube.rs` ; elle n'est PAS
//! l'unique source des plafonds : balayages, chaînes, nombre de β et répliques sont aussi codés dans `cli.rs`, et la note les
//! recopie à la main (concordance relevée à la relecture c16c, pas imposée par le code).

use crate::commun::{Erreur, Resultat};

/// Une ligne du domaine admis : option, commandes concernées (`gen` = `su2 gen --cotes`, `smin`, `apparie`), bornes
/// INCLUSIVES, nature (entier ou réel fini), intervalle lisible et raison.
pub struct Borne {
    pub option: &'static str,
    pub commandes: &'static [&'static str],
    pub min: f64,
    pub max: f64,
    pub entier: bool,
    pub intervalle: &'static str,
    pub raison: &'static str,
}

const TOUTES: &[&str] = &["gen", "smin", "apparie"];
const GEN: &[&str] = &["gen"];
const SMIN: &[&str] = &["smin"];
const APPARIE: &[&str] = &["apparie"];
const GEN_APPARIE: &[&str] = &["gen", "apparie"];

/// 2³² − 1 : nombre maximal de balayages de bain thermique par chaîne (numéro de balayage Philox sur 32 bits).
pub const BALAYAGES_MAX: u64 = u32::MAX as u64;
/// Volume maximal (sites) : 4V < 2³² et au plus ~3,6 Gio de champ et de tables par chaîne.
pub const VOLUME_MAX: u64 = 1 << 24;

/// La table du domaine admis. Contraintes croisées (vérifiées au même endroit) : volume ℓ0ℓ1ℓ2ℓ3 ≤ 2²⁴ ;
/// therm + nmeas·skip ≤ 2³² − 1 ; `--dump-every` ≤ nmeas ; période N ≤ nmeas avec ⌊nmeas/N⌋ ≥ 2 ; R ≤ ℓ_a/2, T ≤ ℓ_b/2 ;
/// `--nmeas` d'apparie multiple de 20 ; `--bruit` > 0 depuis les départs stationnaires mince et abelien ; graines des chaînes
/// d'un lot deux à deux distinctes.
pub const DOMAINE: &[Borne] = &[
    Borne {
        option: "--cotes",
        commandes: TOUTES,
        min: 2.0,
        max: 4096.0,
        entier: true,
        intervalle: "chaque côté entier dans [2, 4096]",
        raison: "tore périodique ; avec le volume, garde tables et estimations de mémoire représentables",
    },
    Borne {
        option: "volume",
        commandes: TOUTES,
        min: 16.0,
        max: 16_777_216.0,
        entier: true,
        intervalle: "ℓ0ℓ1ℓ2ℓ3 dans [2⁴, 2²⁴] sites",
        raison: "4V < 2³² (indice de lien du compteur Philox) ; au plus ~3,6 Gio de champ et de tables par chaîne",
    },
    Borne {
        option: "--beta",
        commandes: GEN_APPARIE,
        min: 1e-3,
        max: 1e3,
        entier: false,
        intervalle: "réel fini dans [10⁻³, 10³]",
        raison: "couplages du module (production : 1 à 48) ; sous 10⁻³, α = β|A| → 0 où le tirage partagé de x₀ peut épuiser \
                 ses 10⁶ essais ; au-dessus, marge contre les débordements de β·V",
    },
    Borne {
        option: "nombre de β",
        commandes: APPARIE,
        min: 1.0,
        max: 256.0,
        entier: true,
        intervalle: "1 à 256 valeurs",
        raison: "tâches β × répliques × secteurs bornées (clés de tâche k < 2¹⁹)",
    },
    Borne {
        option: "--therm",
        commandes: GEN,
        min: 0.0,
        max: 4_294_967_294.0,
        entier: true,
        intervalle: "entier dans [0, 2³² − 2]",
        raison: "therm + nmeas·skip ≤ 2³² − 1 balayages par chaîne (compteur Philox des balayages sur 32 bits)",
    },
    Borne {
        option: "--therm",
        commandes: APPARIE,
        min: 0.0,
        max: 4_294_967_275.0,
        entier: true,
        intervalle: "entier dans [0, 2³² − 21]",
        raison: "therm + nmeas·skip ≤ 2³² − 1 balayages par chaîne (compteur Philox des balayages sur 32 bits)",
    },
    Borne {
        option: "--nmeas",
        commandes: GEN,
        min: 1.0,
        max: 4_294_967_295.0,
        entier: true,
        intervalle: "entier dans [1, 2³² − 1]",
        raison: "au moins une mesure ; therm + nmeas·skip ≤ 2³² − 1",
    },
    Borne {
        option: "--nmeas",
        commandes: APPARIE,
        min: 20.0,
        max: 4_294_967_280.0,
        entier: true,
        intervalle: "multiple de 20 dans [20, 2³² − 16]",
        raison: "moyenne et erreur sur les mêmes 20 blocs égaux ; therm + nmeas·skip ≤ 2³² − 1",
    },
    Borne {
        option: "--skip",
        commandes: GEN,
        min: 1.0,
        max: 4_294_967_295.0,
        entier: true,
        intervalle: "entier dans [1, 2³² − 1]",
        raison: "au moins un pas entre deux mesures ; therm + nmeas·skip ≤ 2³² − 1",
    },
    Borne {
        option: "--skip",
        commandes: APPARIE,
        min: 1.0,
        max: 214_748_364.0,
        entier: true,
        intervalle: "entier dans [1, ⌊(2³² − 1)/20⌋ = 214748364]",
        raison: "au moins un pas entre deux mesures ; therm + nmeas·skip ≤ 2³² − 1 avec nmeas ≥ 20",
    },
    Borne {
        option: "--nor",
        commandes: GEN_APPARIE,
        min: 0.0,
        max: 100.0,
        entier: true,
        intervalle: "entier dans [0, 100]",
        raison: "sur-relaxations par pas (production : 1 à 4) ; au-delà, coût sans usage",
    },
    Borne {
        option: "--chaines",
        commandes: GEN,
        min: 1.0,
        max: 4096.0,
        entier: true,
        intervalle: "1 à 4096 chaînes (listes --beta et --graine comprises)",
        raison: "un lot borné ; graines des chaînes deux à deux distinctes",
    },
    Borne {
        option: "--graine",
        commandes: TOUTES,
        min: 0.0,
        max: u64::MAX as f64,
        entier: true,
        intervalle: "entier dans [0, 2⁶⁴ − 1]",
        raison: "toute graine 64 bits : clé Philox = graine (gen, smin) ou graine + 7919·k (apparie)",
    },
    Borne {
        option: "--graine-pas",
        commandes: GEN,
        min: 0.0,
        max: u64::MAX as f64,
        entier: true,
        intervalle: "entier dans [0, 2⁶⁴ − 1], graines du lot distinctes",
        raison: "graine + i·pas modulo 2⁶⁴ ne doit retomber sur aucune graine du lot",
    },
    Borne {
        option: "--dump-every",
        commandes: GEN,
        min: 0.0,
        max: 4_294_967_295.0,
        entier: true,
        intervalle: "0 (jamais) ou entier dans [1, nmeas]",
        raison: "configurations écrites comptées dans la projection disque et la mémoire du catalogue",
    },
    Borne {
        option: "--mesures tube:…:N (période)",
        commandes: GEN,
        min: 1.0,
        max: 2_147_483_647.0,
        entier: true,
        intervalle: "entier dans [1, ⌊nmeas/2⌋] (au plus 2³¹ − 1)",
        raison: "au moins deux mesures d'observables (moyenne et erreur de blocs)",
    },
    Borne {
        option: "--mesures tube:ab:RMAX",
        commandes: GEN,
        min: 1.0,
        max: 2048.0,
        entier: true,
        intervalle: "entier dans [1, ℓ_a/2]",
        raison: "boucles de Wilson sans recouvrement sur le tore",
    },
    Borne {
        option: "--mesures tube:ab:RMAX:TMAX",
        commandes: GEN,
        min: 1.0,
        max: 2048.0,
        entier: true,
        intervalle: "entier dans [1, ℓ_b/2]",
        raison: "boucles de Wilson sans recouvrement sur le tore",
    },
    Borne {
        option: "--gtol",
        commandes: SMIN,
        min: 1e-14,
        max: 1e-2,
        entier: false,
        intervalle: "réel fini dans [10⁻¹⁴, 10⁻²]",
        raison: "sous 10⁻¹⁴, plancher d'arrondi du gradient en F64 ; au-dessus de 10⁻², ce n'est plus un point stationnaire",
    },
    Borne {
        option: "--bruit",
        commandes: SMIN,
        min: 0.0,
        max: std::f64::consts::TAU,
        entier: false,
        intervalle: "réel fini dans [0, 2π], > 0 pour --depart mince ou abelien",
        raison: "angle de rotation ; mince et abelien sont stationnaires : sans bruit le refroidissement n'en bouge pas",
    },
    Borne {
        option: "--balayages",
        commandes: SMIN,
        min: 0.0,
        max: 4_294_967_295.0,
        entier: true,
        intervalle: "entier dans [0, 2³² − 1]",
        raison: "plafond du refroidissement, exact (dernier paquet tronqué)",
    },
    Borne {
        option: "--repliques",
        commandes: APPARIE,
        min: 1.0,
        max: 1024.0,
        entier: true,
        intervalle: "entier dans [1, 1024]",
        raison: "tâches bornées ; clés de tâche k = ib·max(1000, 2·répliques) + 2·réplique + secteur < 2¹⁹",
    },
];

fn ligne(commande: &str, option: &str) -> Resultat<&'static Borne> {
    DOMAINE
        .iter()
        .find(|b| b.option == option && b.commandes.contains(&commande))
        .ok_or_else(|| Erreur::Config(format!("domaine : pas de borne pour {option} ({commande})")))
}

fn hors(b: &Borne, valeur: &str) -> Erreur {
    Erreur::Config(format!(
        "{} {valeur} : hors du domaine admis ({}) — {}",
        b.option, b.intervalle, b.raison
    ))
}

/// Vérifie une valeur déjà calculée (volume, nombre de valeurs, élément d'une liste…) contre la ligne (commande, option).
pub fn borne(commande: &str, option: &str, x: f64) -> Resultat<()> {
    let b = ligne(commande, option)?;
    if x.is_finite() && (b.min..=b.max).contains(&x) && (!b.entier || x.fract() == 0.0) {
        Ok(())
    } else {
        Err(hors(b, &format!("{x}")))
    }
}

/// Réel lu de `texte` : fini et dans l'intervalle de (commande, option), sinon refus d'usage.
pub fn reel(commande: &str, option: &str, texte: &str) -> Resultat<f64> {
    let b = ligne(commande, option)?;
    let x: f64 = texte.trim().parse().map_err(|_| hors(b, texte))?;
    if x.is_finite() && (b.min..=b.max).contains(&x) {
        Ok(x)
    } else {
        Err(hors(b, texte))
    }
}

/// Entier lu de `texte` (u64) dans l'intervalle de (commande, option), sinon refus d'usage. Les bornes entières de la table
/// sont exactes en f64 (≤ 2³²), ou valent `u64::MAX as f64` = 2⁶⁴ pour les graines, qui admettent tout u64.
pub fn entier(commande: &str, option: &str, texte: &str) -> Resultat<u64> {
    let b = ligne(commande, option)?;
    let n: u64 = texte.trim().parse().map_err(|_| hors(b, texte))?;
    if (b.min..=b.max).contains(&(n as f64)) {
        Ok(n)
    } else {
        Err(hors(b, texte))
    }
}

/// Le domaine admis de `commande`, en texte (aide `--help` et note).
pub fn table_aide(commande: &str) -> String {
    let mut s =
        format!("Domaine admis ({commande}) ; hors domaine : refus, code 2, rien d'écrit :\n");
    for b in DOMAINE.iter().filter(|b| b.commandes.contains(&commande)) {
        s.push_str(&format!(
            "  {} : {} — {}\n",
            b.option, b.intervalle, b.raison
        ));
    }
    s
}
