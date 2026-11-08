# validation/wp9 — campagne pilote « couplage effectif, un pas de bloc » (WP9)
PREREG : `prereg/WP9_PREREG.md` (gelé) et avenant `prereg/WP9_AVENANT_AV1.md`. Sorties de la campagne (hors dépôt) :
`kevinotron_unification/campagnes/wp9/` (flot.json par cellule, analyse, VERDICT.md).
- `analyse_wp9.py` : règles de décision du PREREG §6 (C_cal par rééchantillonnage des courbes, MC sous le nul, σ_stat = max(σ_K, σ_2K),
  GLS ; R2 domaine, R4 localité, R5 réplicats, R6 deux machines, ajustement descriptif). Usage :
  `python analyse_wp9.py --cellules cellules_wp9.json --courbe4 etalonnages/etal_L4_wp9.json --courbe8 etalonnages/etal_L8_wp9.json --sortie DIR`
  (chemins des flot.json relatifs au fichier de cellules). Testé contre le binaire : WP9-ANALYSE-INV (PCHIP scipy = PCHIP du binaire à 1e-14).
- `comparer_flot.py` : deux flot.json (backends ou machines différents) à ≤ 1e-9 relatif, compteurs de domaine identiques (WP9-FLOT-GPU, WP9-2MACH).
- `comparer_analyses.py` : deux analyse.json à ≤ 1e-12 (rejeu gamer contre acemagic).
- `etalonnages/` : courbes `rg etalonner --protocole wp9` (L = 4 et 8 ; 31 β ; graines 26102620000 / 26102624000 + 100 i + c), chaînes
  indépendantes (c = 9), mutant 1,05 β (L = 4), chaînes aux milieux (c = 8). **RG-PLAQ-WP9-MILIEU est FAIL** sur ces courbes (biais de PCHIP
  aux sauts de pas de la grille) : elles ne doivent pas servir à une mesure plaq/mcrg2 sans nouvel étalonnage (AV1 §4).
