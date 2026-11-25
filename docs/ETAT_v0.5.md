# kvu v0.5 : état, carte des modules, backends validés, limites

Branche `integration/v0.5` (23/11/2026, fusion finale) = `integration/v0.4` plus `geometrie/coutures-gpu-1122`, les branches de Codex (orbites regroupées, flux anisotrope et secteur (1,1,1), transport), les dettes de la fusion de nuit (`sprint_2026-11-22/fusion-4/RAPPORT.md`) et `thermo/couplages-separes-1122`.
Le numéro de version du paquet est 0.5.1 : `kvu --version` affiche `kvu 0.5.1` (la 0.5.0 est la version interne non publiée sur laquelle ce document a été écrit ; la 0.5.1 y ajoute les correctifs d'arrêt brutal PUB-14 et PUB-16 et des corrections de style de clippy). Cette version est celle de la publication 0.5.1 (voir `docs/PUBLICATION.md`).
Les modules déjà présents en v0.3 et v0.4 sont inchangés : voir `docs/ETAT_v0.3.md` et `docs/ETAT_v0.4.md` pour leur carte.

## Mauvaises nouvelles
- kvu mesure, il ne démontre rien : aucun module ne prouve le trou de masse.
- Impulsions hors axe : la cible de Codex (deux orbites <= 1,2 fois le cas sans impulsion) n'est PAS atteinte ; rapports mesurés dans `fusion-4/RAPPORT.md` (HIP 1,234, CUDA 1,340, mieux que v0.4). La cible de temps des orbites sur CUDA reste ouverte.
- Lien intégré : le noyau coûte du temps de mesure (sur la Radeon : +45 % en F64, +76 % en F32) ; à couplage fort (beta >= 1,5) le gain de variance est inférieur au surcoût. Il sert surtout aux petits couplages temporels (`thermo-aniso/RAPPORT.md`, tableau de gain).
- Lien intégré : S_feuille (torsion) à lien intégré NON fait ; refusé avec torsion, torsion fixe et couture. `--beta-s` refusé avec torsion, torsion fixe, `--poids` différent de 1 et couture décalée.
- `su2 thermo analyse` (modes torsion et a3) refuse les dossiers `--beta-s` : l'intégration en beta temporel est à faire à part (colonnes `st`, `y0_t`, `ys_t`).
- Les poids de la combinaison optimale {brut, Y0, Ys} sont estimés sur les mêmes répliques (biais en 1/R non borné).
- Transport (Codex) : outils Python seulement, pas de vérification sur une campagne réelle complète.
- Les cartes AMD et NVIDIA ne donnent pas les mêmes chiffres au bit (arrondis) ; les moyennes sont compatibles.
- `intrication_cli` et trois autres suites dépendent de la place disque : `KEVINOTRON_SSD_MIN_GO=0` quand `/tmp` a moins de 100 Go libres.

## Carte des outils ajoutés depuis v0.4 (éteints par défaut)
| Outil | Option | cpu | amd | nvidia |
|---|---|---|---|---|
| Couture décalée sur carte | `su2 thermo mesure --decalage v0,v1,v2,v3 [--decalage-b ...] [--decalage-axe A]` | gen seulement | oui | oui |
| Impulsions hors axe regroupées | `su2 spectro gpu --impulsions-vecteurs 2:2:1,3:0:0` | oui | oui | oui |
| Flux anisotrope, secteur (1,1,1) | `su2 spectro gpu-torelon --secteur-flux 1,1,1` | référence dans les tests | oui | oui |
| Couplages séparés | `su2 thermo mesure --beta-s BS --betas bt1,bt2,...` | non | oui | oui |
| Lien intégré (Y0, Ys, Zs) | `su2 thermo mesure --lien-integre` | non | oui | oui |
| Séries de torsion (lecteur corrigé) | `scripts/.../lire_series.py` | script | script | script |
| Générateur et lecteur de transport | `python3 -I scripts/thermo/transport.py` (voir `scripts/thermo/TRANSPORT.md`) | script | script | script |

## Ce qui est vérifié
Les nombres de la qualification (tests, identité au bit, cartes) sont dans `sprint_2026-11-22/fusion-5/RAPPORT.md`. Résumé des sources de la nuit : `fusion-4/RAPPORT.md` (coutures sur carte, orbites, flux (1,1,1), lecteur, test G4 à plusieurs groupes de beta) et `thermo-aniso/RAPPORT.md` (couplages séparés, lien intégré : carte contre hôte à 1e-15 en F64, contre numpy à 1e-16).

## Ce qui manque
- S_feuille à lien intégré ; `analyse` en anisotrope ; cible de temps des orbites sur CUDA (1,2 non atteint) ; couture décalée avec `--beta-s` et `--lien-integre`.
- Optimisation du noyau de lien intégré ; voie processeur pour `thermo mesure`.
