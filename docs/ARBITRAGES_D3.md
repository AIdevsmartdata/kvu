# Arbitrages D3 — les trois gates rouges de l'intégration (lot WP2)

Décision de Kévin du 25/10/2026 (D3, « oui à tout » aux défauts proposés) ; seuils gelés dans `GATES.md` v1 (sha256
`c1eb4f9f…5163`, journal `WP0`), §1/WP2. **Aucun autre seuil n'a été modifié.** Le rejeu complet a été fait d'abord, sans
modification, sur les binaires figés de WP1 (phase A) ; les arbitrages ont été appliqués ensuite (phase B), chacun dans son
commit.

| gate | ancien seuil | nouveau seuil | raison chiffrée | renvoi |
|---|---|---|---|---|
| OBS-1 | `max |ΔW|/max(|W|,…) ≤ 1e-13` (relatif) sur les tables Wilson GPU contre CPU | `max |ΔW| ≤ 1e-15` (**absolu**) sur toutes les tables ; autres volets inchangés (champ froid W = 1 exactement, invariance de jauge ≤ 1e-13, F32 promu, déterminisme) ; **mutant** : +1e-14 sur un W de la référence CPU ⇒ OBS-1 ROUGE (gate `OBS1-MUTANT`) | W est une moyenne de termes ½tr ∈ [−1, 1] ; un ordre de sommation différent (arbre fixe GPU, boucle CPU) donne un écart absolu ~ ε et un écart relatif ~ ε/\|W\|, qui diverge quand W → 0 : pire cas W(5,4) = −1,26e-5, écart absolu 3,5e-18, relatif 2,751e-13 (rouge aux deux machines) ; le pire écart absolu de toutes les tables était 1,1e-16 | GATES v1 §1/WP2 « OBS-1 (v-kvu) » ; RAPPORT_INTEGRATION §4.1 |
| OBS-4 | « rapportée, critère souple ≥ 90 % » à L = 8 (comptée FAIL par l'orchestrateur : 15/20) | L = 8 reclassé **INFO** (`OBS-4-L8`, ni PASS ni FAIL) ; nouvelle gate **`OBS-4-L16`** : L = 16, β = 2,5, 20 configurations d'une chaîne (thermalisation 2000, espacées de 100 pas), flot t = 3 (dt = 0,02) : **≥ 18/20** avec \|Q − round(Q)\| < 0,1 | à L = 8 la charge « clover » non améliorée n'est pas quantifiée à 0,1 près (√(8t) ≈ 4,9 > L/2) : 15/20, valeurs 0,859 ; 0,868 ; 0,755 ; 0,853 ; 0,215 pour les cinq autres ; OBS-3 montre Q GPU = Q CPU à 3,9e-16 (le code calcule ce qui est demandé) | GATES v1 « OBS-4-L8 », « OBS-4-L16 » ; RAPPORT_INTEGRATION §4.2 |
| G-H11 | C1-C4 de PREREG_CRITERES tous verts, sinon FAIL | cellule (L = 8, β = 3,5, C2 chaud/froid) : statut **FAIL-DOCUMENTÉ** (non bloquant, listé) **seulement si** z = −3,28, seul critère rouge, et chaînes de graines 27092708021/22/23 identiques à l'octet (`measurements.csv` et `therm.csv`) aux preuves ; tout autre z ou sha ⇒ **FAIL réel (régression)** ; jamais relancée avec d'autres graines | fait statistique du protocole (C2 sur 2000 pas de thermalisation à β = 3,5), pas une régression : les 3 chaînes sont identiques à l'octet à `su2_hip` lancé seul (22 000 pas), z = −3,28 | GATES v1 « G-H11-DOC » ; RAPPORT_INTEGRATION §4.3 ; `validation/integration/preuves/gh11_bit_identite.txt` |

## Mise en œuvre

- `crates/kvu/tests/obs_gpu.rs` : `SEUIL_OBS1_ABS = 1e-15`, juge `juge_obs1` NaN-sûr (un NaN donne +∞ ; l'ancien `fold(0, f64::max)` l'ignorait),
  test `obs_1_mutant_rouge` (imprime `PASS OBS1-MUTANT` quand le mutant est bien détecté ROUGE, `FAIL … ECHEC_OUTIL` sinon),
  `obs_4_topologie_apres_flot_rapport` → ligne `INFO OBS-4-L8`, nouveau test `obs_4_l16_quantification` (graine 26102516001 fixée
  avant tout run, générateur GPU F64 compat-prod, 1 bain thermique + 4 OR par pas).
- `crates/kvu/src/cli/catalogue.rs` : statuts `INFO` et `FAIL-DOCUMENTE` (ni PASS, ni FAIL, ni SKIP ; le code de sortie ne les compte pas,
  le tableau et le JSON les listent) ; jetons `OBS-4-L8`, `OBS-4-L16`, `OBS1-MUTANT` à la place de `OBS-4`.
- `validation/gates_unifie.sh`, `validation/obs/gates_obs.sh`, `validation/historique/G-H11.sh` : prise en charge des deux statuts ; G-H11 compare
  par sha256 les chaînes aux preuves avant d'accorder `FAIL-DOCUMENTE`.

## Résultats après arbitrage (rejeu sur le commit `74cd275`, binaires de test rebâtis ; `kvu` WP1 inchangé)

| gate | acemagic (HIP) | gamer (CUDA) | détail |
|---|---|---|---|
| OBS-1 | **PASS** | **PASS** | écart absolu max 1,11e-16 ≤ 1e-15 (L = 6 : 5,55e-17 ; F32 promu : 5,55e-17 ; jauge : 1,11e-16 ; champ froid exact) |
| OBS1-MUTANT | **PASS** (mutant détecté ROUGE) | **PASS** | +1e-14 sur W[15] de la référence CPU ⇒ écart 9,99e-15 > 1e-15 ; référence propre 1,11e-16 |
| OBS-1-HIST | **PASS** | — | `avg_wilson_loop` historique (extraction `git archive 0a7a2de`, cible dédiée) : 1,22e-15 ≤ 1e-12 |
| OBS-4-L8 | INFO (15/20) | INFO (15/20) | rapportée, ne verdit pas |
| OBS-4-L16 | **FAIL réel : 5/20** | **FAIL réel : 5/20** | seuil gelé ≥ 18/20 ; graine 26102516001 posée avant tout run, non changée ; voir ci-dessous |
| G-H11 | **FAIL-DOCUMENTÉ** | — | z = −3,28 ; seul critère rouge L8b3.5:C2_ok ; `measurements.csv` et `therm.csv` des chaînes 27092708021/22/23 identiques à l'octet à `su2_hip` de production lancé seul (sha256 `37fb932b…`, `a7b5621e…`, `cb24da48…` pour measurements = `gh11_bit_identite.txt` §b ; therm : `preuves/gh11_therm_wp2.txt`) ; non relancée |
| G-H04-HIP ×2 | PASS, PASS | — | deux exécutions consécutives, mêmes graines, lignes identiques ; plus le PASS de la phase A : pas d'ALERTE-IGPU |

### OBS-4-L16 : le critère D3 n'est pas atteint (constat, pas un réglage)
Le critère « ≥ 18/20 à moins de 0,1 d'un entier à t = 3 » est **rouge sur les deux machines** (5/20 chacune), alors que le générateur
est sain (plaquette moyenne des 20 configurations 0,652, entre 0,602 à β = 2,3 et 0,6995 à β = 2,8 ; OBS-3 : Q GPU = Q CPU à 4e-16).
Diagnostic **hors gate** (`obs_4_l16_diagnostic_flot_long`, lignes `INFO OBS-4-L16-DIAG`, ajouté après le verdict rouge, mêmes 20
configurations) : nombre de charges à moins de 0,1 d'un entier à t = 3 ; 4,5 ; 6 ; 9 ; 12 = 5, 3, 6, 8, 9 sur 20 (acemagic) et 5, 3, 5, 3, 5
sur 20 (gamer). Les charges se stabilisent à des valeurs non entières (≈ 0,90 ; 1,82 ; 1,77 ; 0,95…) : la charge « clover » **non améliorée**
sous-estime la charge de dislocations de réseau à β = 2,5, et un flot plus long ne l'entier-ise pas. Conclusion pour l'arbitre : la
voie « L = 16 » de D3 ne suffit pas ; il faut une définition améliorée de Q (ou un autre critère, ex. écart à l'entier le plus proche
avec un facteur de renormalisation) **avant** TOPO-Q (WP19, même seuil). Aucun seuil n'a été changé ici : OBS-4-L16 reste FAIL.
