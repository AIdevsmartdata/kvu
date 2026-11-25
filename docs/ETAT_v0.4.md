# kvu v0.4 : état, carte des modules, backends validés, limites

Branche `integration/v0.4` (22/11/2026, fusion n°3) = `integration/v0.3` plus quatre outils : coutures décalées, impulsions hors axe, secteur de flux (1,1,0), séries et torsion fixe.
Le numéro de version du paquet est passé de 0.1.0 à 0.4.0 : `kvu --version` affiche maintenant `kvu 0.4.0`. Elle est incluse dans la publication 0.5.0 (voir `docs/PUBLICATION.md`).
Les modules déjà présents en v0.3 sont inchangés : voir `docs/ETAT_v0.3.md` pour leur carte complète.

## Mauvaises nouvelles
- kvu mesure, il ne démontre rien : aucun module ne prouve le trou de masse.
- Coutures décalées (tores obliques) : processeur seulement. Pas de carte, pas de `thermo`, pas de torsion, pas d'observables du tube.
- Secteur de flux : l'opérateur du secteur (1,1,1) n'existe pas (refusé). Le fichier de corrélations est lourd ; pour beaucoup de configurations, utiliser `--flux-sans-corr`.
- `thermo mesure` (séries, torsion fixe) : pas de voie processeur, la carte est obligatoire. Le contrôle de bout en bout des torsions est faible (1,5 écart type).
- Impulsions hors axe : surcoût sur la carte (2 à 4 fois le temps d'une mesure sans impulsion, selon le nombre d'orbites). La reprise (`--reprendre`) n'était pas testée sur un vrai redémarrage.
- Les cartes AMD et NVIDIA ne donnent pas les mêmes chiffres au bit (arrondis) ; les moyennes sont compatibles.
- Aucune des options ne gagne de portée par la fusion. Le test `cmp` de l'option de flux n'est pas dans la suite cargo (il exige une carte).
- `intrication_cli` dépend de la place disque : la garde SSD refuse d'écrire sous 100 Go libres ; les tests passent avec `KEVINOTRON_SSD_MIN_GO=0`.
- `rg gates-gpu` : 3 portes sautées et un KO connu (fichiers de référence absents).

## Carte des quatre outils (éteints par défaut)
| Outil | Option | cpu | amd | nvidia |
|---|---|---|---|---|
| Couture décalée | `su2 gen --decalage v0,v1,v2,v3 [--decalage-axe A]` | oui | non | non |
| Impulsions hors axe | `su2 spectro gpu|neutre --impulsions-vecteurs 2:2:1,3:0:0` | oui | oui | oui |
| Secteur de flux (1,1,0) | `su2 spectro gpu-torelon --secteur-flux 1,1,0 [--flux-sans-corr]` | référence dans les tests | oui | oui |
| Séries de torsion | `su2 thermo mesure --series` | non | oui | oui |
| Torsion fixe | `su2 thermo mesure --torsion-fixe 01 --torsion 02` | non | oui | oui |

## Ce qui est vérifié (chiffres de `sprint_2026-11-22/fusion-3/RAPPORT.md`)
- Chemins existants (`gen`, `spectro neutre` et `gpu`, `gpu-torelon`, `thermo mesure`) : fichiers identiques au bit à ceux de v0.3 sur processeur, AMD et NVIDIA, options éteintes.
- Options : le binaire fusionné donne les mêmes fichiers au bit que le binaire de chaque branche (impulsions hors axe, secteur de flux, séries et torsion fixe sur AMD et NVIDIA ; coutures sur processeur).
- Tests processeur : 1 833 réussis, 0 échec, 197 ignorés (tests de carte, à lancer à part avec leur verrou).
- Essais de fumée : 33 OK et 1 KO sur AMD, 33 OK et 1 KO sur NVIDIA (le KO est `rg gates-gpu`) ; processeur 27 OK et 7 refus attendus (commandes qui exigent une carte).

## Ce qui manque
- Coutures sur carte ; plusieurs coutures non diagonales ; test exact en 4D pleine.
- Secteur (1,1,1) ; plusieurs groupes de β avec `--series` ; voie processeur pour `thermo mesure`.
- Optimisation du noyau des orbites hors axe.
