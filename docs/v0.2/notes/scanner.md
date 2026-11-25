# Fragment v0.2 : module scanner (`kvu obs champs`), à assembler par l'agent de documentation

## Ce que fait le module
`kvu obs champs --cfg X.npy --out DIR` exporte sept champs par site d'UNE configuration SU(2) (action brute et après flot,
énergie et charge au trèfle après flot, Polyakov dans les 4 directions, nombres de plaquettes à trace négative brut et après
flot) en NPY (f32 ou f64) avec le manifeste commun `manifest.json` (format `kvu-champs/1`). Calcul CPU FP64 sur la référence
`obs::ref_cpu`, aucun nouvel algorithme de physique. Détail : `docs/SCANNER_CHAMPS.md`. Fichiers : `obs/champs.rs`, `obs/champs_cli.rs`,
`tests/obs_champs.rs`. Ajouts aux fichiers partagés : deux lignes de `obs/cli.rs`, trois de `obs/mod.rs`, une rubrique de `GUIDE.md`.

## Portes
CH-SOMME-* (onze : chaque agrégat contre l'observable globale existante de kvu), CH-LOCALITE (64 sites, chemins indépendants),
CH-JAUGE-* (sept, `--porte-jauge`). Chaque porte a un mutant rouge dans `tests/obs_champs.rs`. Le code de retour est 1 si une
porte est rouge (pas de COMPLETE).

## Limites
- Configuration périodique SANS torsion seulement ; torsion ou bord déclarés dans `torsion.json` / `manifest.json` /
  `config_run.json` (dossier de la cible et son parent) sont refusés ; sans métadonnée, la périodicité est SUPPOSÉE (consigné).
- `plaq_negatives_*` est le signe de la trace des plaquettes, pas une projection de centre (pas de P-vortex).
- La direction du temps n'est que consignée. Une image tirée de ces champs ne prouve pas un trou de masse.
- Porte de jauge : plancher O(1) du dénominateur (une non-invariance < 1e-10 sur un champ < 1 n'est pas vue).

## Réponse à la relecture de GPT-6.1 Sol (11/11/2026, `ORCHESTRE/revues/c01-scanner/revue.md`, NON GELABLE)
| objection | verdict | correction |
|---|---|---|
| O1 refus contourné par un chemin relatif | fondée, corrigée | chemin absolu + résolu avant la recherche des métadonnées |
| O2 `f64::max` ignore les NaN | fondée, corrigée | `max_nan`, écart NaN si non fini, refus d'un export non fini (même `--sans-portes`) |
| O3 nombre de pas saturé | fondée, corrigée | refus au-delà de 2^53 pas ou temps non fini |
| O4 porte de jauge sur champ nul | fondée (écart 2e-14 mesuré sur le champ froid), corrigée | plancher O(1) du dénominateur |
| O5 backend du manifeste | fondée, corrigée | `backend = "cpu"`, `arch = null`, `precision_calcul = "FP64"` |
| O6 tests dépendants de la garde SSD | fondée, corrigée | essais CLI en sous-processus isolé |
| O7 trèfle brut jeté | fondée, corrigée | la phase brute ne calcule plus le trèfle (champs identiques à l'octet) |

Non-régression : pour une entrée valide les sept fichiers de champs sont identiques à l'octet entre f9ff3b5 et la tête (3
variantes : f32 avec jauge, f64, sans portes). Seules les métadonnées du manifeste changent (chemins `fichiers_lus` absolus, détail et écart
des portes de jauge, `precision_calcul`, backend/arch, empreintes et durées).
