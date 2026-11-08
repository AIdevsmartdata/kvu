# Défauts connus de kvu 0.1.0

## 1. Charge topologique de `obs flot` : non quantifiée à β = 2,5 (gate OBS-4-L16 déclassée)

**Statut** : observable **expérimental, défaut connu**. La gate `OBS-4-L16` (test GPU `obs_gpu::obs_4_l16_quantification`)
échoue sur les deux backends (passe GPU du 8 novembre 2026) : **5/20** configurations sur CUDA (RTX 5060 Ti ; rejeu identique au
bit) comme sur HIP (Radeon 8060S ; la chaîne générée diffère d'un backend à l'autre) ont `|Q − round(Q)| < 0,1` après un flot
de Wilson jusqu'à t/a² = 3 (L = 16, β = 2,5, 20 configurations d'une chaîne, graine 26102516001), pour un seuil gelé de
**18/20**. Elle est **déclassée** (échec documenté, non bloquant) pour la publication 0.1.0 ; ni le test ni son seuil n'ont été
modifiés, et elle reste rouge.

**Ce que mesure la gate.** `kvu obs flot` donne `Q_top` par la définition « trèfle » (clover) non améliorée,
Q = −(1/(32π²)) Σ_x ε_{μνρσ} tr(G_μν G_ρσ) (`crates/kvu/src/obs/ref_cpu.rs`), à chaque mesure du flot.

**Pourquoi ce n'est pas une erreur de normalisation, de signe ou de flot** :

- même calcul sur GPU et sur la référence CPU : écart 4e-16 (gate OBS-3) ;
- normalisation : sur un instanton discrétisé, la charge trèfle vaut 1 à 4e-4 près à L = 16 (gate TOPO-INST,
  `crates/kvu/validation/topo/RENDU_WP19.md`) ; après 20 balayages de cooling, ⟨Q_L²⟩ (trèfle non arrondi) = 6,702 contre
  6,635 ± 0,056 pour Athenodorou et Teper à β = 2,427 sur 16⁴ (`crates/kvu/validation/topo/topo.json`). Une erreur de facteur k
  donnerait k² ; le rapport est 1,01 ;
- un signe faux n'empêcherait pas la quantification ; le flot n'est pas en cause : `obs topo` emploie le même pas RK3 de
  Lüscher (`pas_rk3` de `ref_cpu.rs`) et sa charge améliorée, calculée sur les configurations flotées, se quantifie (ci-dessous) ;
- prolongé jusqu'à t/a² = 12, le flot ne rend pas les charges entières : la plupart se stabilisent à environ 0,86 à 0,96 fois
  un entier, **en dessous** de cet entier en valeur absolue (−0,88 ; 0,93 ; −1,79 ; 1,73 ; −2,69 ; −2,82 ; 3,79…), et
  quelques-unes dérivent encore (1,11 → 1,29 ; −1,64 → −0,69) ; diagnostic hors gate `obs_4_l16_diagnostic_flot_long`, lignes
  `INFO OBS-4-L16-DIAG` (journaux de la passe GPU du 8 novembre 2026).

**Explication.** C'est l'effet de discrétisation connu de la charge non améliorée. Athenodorou et Teper (arXiv:2106.00364v3,
§ sur la topologie, commentaire de leur figure 20) écrivent que le pic des charges refroidies n'est pas sur un entier parce que
« for an instanton of size ρ on a lattice, we only obtain Q_L → 1 in the limit ρ/a → ∞, with deviations from unity that are
powers of a/ρ », et que « the discretisation corrections are negative ». Ils attribuent une charge entière Q_I à chaque
configuration plutôt que d'exiger Q_L entier. Le critère « 0,1 d'un entier pour 90 % des configurations, à t/a² = 3, avec la
charge trèfle » est donc plus strict que ce que cette définition donne à β = 2,5 ; aucun seuil n'a été changé pour autant.

**Ce qui marche.** La charge **améliorée** de `kvu obs topo` (module WP19) se quantifie : 70/100 à t/a² = 3, 89/100 à 5,
95/99 à 2 t₀ et 98/99 à 3 t₀, contre 30/100 à 49/99 pour le trèfle sur les mêmes configurations (`RENDU_WP19.md`, lignes
`TOPO-Q-T`). Pour une charge entière, utiliser `kvu obs topo` et un temps de flot exprimé en t₀ ; le `Q_top` de `obs flot` est
une sortie de contrôle.

**Suite prévue (0.2)** : remplacer, dans `obs flot`, la charge trèfle par la charge améliorée (ou ajouter un arrondi
documenté), et reformuler la gate en conséquence, avec un nouveau critère gelé avant tout calcul.

## 2. Gates et tests GPU qui ne tournent qu'avec leur script de validation

- Les tests F32 de `tests/su2_f32_gpu.rs` ne font que `SKIP` hors de `validation/f32/gates_f32.sh` (variable
  `KVU_F32_SOUS_VERROU=1`) ; les gates de loi LOI-0 à LOI-4 sont une campagne pilotée par ce script.
- `G-BIT-K1` demande les références `su2_gen` (`KVU_REF_K1`) ; `MACH-1` et `DETECTION-REELLE` demandent un `kvu.toml` qui
  nomme la machine réelle (la fixture `crates/kvu/kvu.toml` a été neutralisée pour la publication).
