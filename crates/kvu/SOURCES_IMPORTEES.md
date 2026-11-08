# SOURCES_IMPORTÉES — kvu (dépôt autonome, import du 25/10/2026)

## Import du dépôt (WP1)
- Origine : dépôt interne `/chemin/vers/projet/kevinotron`, commit **`0a7a2de`** (branche `unifie/correctifs`, intégration `unifie/integ`) ;
  aucun historique n'est repris (décision D2 : copie, pas de subtree).
- Archive importée : `git -C /chemin/vers/projet/kevinotron archive 0a7a2de crates/kvu src/hip/ffi.rs src/hip/wrap.rs src/su2_hb.rs src/manifest.rs Cargo.lock`
  (sha256 de l'archive : `2e4f039f24ce23154ea58ebedc686a2ba03bbf8c23302d521dc69fe3f6b9053f`) ; l'arbre `crates/kvu` de l'archive est identique à celui de
  `kevinotron_dev/integration/crates/kvu` (`diff -r` vide). `src/hip/su2_sweep.hip` est ajouté à part (`git show 0a7a2de:src/hip/su2_sweep.hip`), car les tests
  `su2_noyaux_sources` le lisent à ce chemin (écart au brief, voir le rendu WP1).
- Écarts de forme par rapport à l'origine : `src/heritage.rs` devient `src/heritage/mod.rs` (mêmes `pub mod`, sans `#[path]`) ;
  `build.rs`, `stockage/provenance.rs`, `stockage/mod.rs` adaptés à la recette `kvu-source/1` ; `Cargo.toml` racine de workspace
  neuf ; `Cargo.lock` réduit aux dépendances du crate par cargo ; `scripts/build_auto.sh` et quelques scripts de `validation/` adaptés.
- Ce fichier existe en deux exemplaires **identiques** (racine et `crates/kvu/`) : G-H02 lit celui de `crates/kvu/`.

# Tableau hérité de l'intégration (branche unifie/base, 04/10/2026)

Copies À L'OCTET, en lecture seule, dans `crates/kvu/reference/`. Elles servent de référence aux
ports (bit-identité, mêmes expressions) et aux gates ; **aucune n'est compilée dans le crate**.
Toute tâche qui en dépend revérifie ces sha256 dans son test (`sha256sum -c` sur ce tableau).

| fichier (reference/…) | origine | rôle |
|---|---|---|
| su2_hip/su2_hip.cpp | /chemin/vers/projet/chantiers/wilson_gpu_grandL_20260926/src/su2_hip.cpp | générateur SU(2) HIP d'origine (source_hash ed1e0fce des ensembles AMD) |
| su2_hip/build.sh | idem, build.sh | options de production : `hipcc --offload-arch=gfx1151 -O3` (contraction par défaut = fast-honor-pragmas) |
| su2_cuda/su2_cuda.cu | gamer:~/ym4_calcul/su2_cuda/su2_cuda.cu | portage mécanique CUDA (5 lignes de diff), sha 510b7fe9 = SHA256SUMS du gamer |
| su2_cuda/hip2cuda.h | gamer:~/ym4_calcul/su2_cuda/hip2cuda.h | correspondance HIP → CUDA (21 #define) |
| su2_cuda/build_cuda.sh | gamer, idem | `nvcc -arch=sm_120 -O3 -std=c++17` (fmad par défaut), cœurs E |
| su2_cuda/test_heatbath_cuda.py, .json | gamer, idem | gate P1 : moments exacts de x₀, 7 α, 2·10⁷ tirages, \|z\| < 4 |
| su2_bord/su2_bord.cu | D6_reduction_R2_20260928/R2_3_influence_bord/calc/su2_bord.cu | dérivé de su2_cuda : bord gelé, jauge du bord, tranches S_t/T_t (binaire 5af3ea49 de S2) |
| su2_bord/tests_code.sh | idem | contrôles C0, C1, C6 (L = 8, β = 2,5 hors grille) |
| fixtures_bord/* | R2_3_influence_bord/calc/tests/{c0_ref,c0_bord,t_cold,t_hot,t_eq,t_eqg} | sorties de su2_bord/su2_cuda (L = 8, β = 2,5, HORS grille de production) ; comparées par `cmp` UNIQUEMENT (aucune analyse de différence de bord) |
| noyau2/cb.py | /chemin/vers/projet/outils/ym4_noyaux/noyau2/cb.py (code v3) | bain conditionnel « fin sachant dec » + mesure de fibre (référence du port T5/T6) |
| noyau2/test_concordance.py, test_statistique.py, README.md, SHA_code_v3.txt | idem | protocoles et seuils des gates de concordance (1e-12) et de loi (EQC-2) |
| gates_gpu1/gates_gpu1.sh, analyse_gpu1.py, RAPPORT_GPU_PORT.md | chantiers/vortex_tests_20260927/gpu_port/ | les 13 gates historiques (s0, s1, P0-P7, runs) à régresser (G-H01..G-H13) |

## sha256 (calculés à la copie ; origines vérifiées : su2_hip.cpp = ed1e0fce…, su2_cuda.cu = 510b7fe9…, su2_bord.cu = 0a01a43e…)

```

```

## Copies héritées du moteur historique (dépôt autonome : copiées À L'OCTET, plus aucun `#[path]`)

Les quatre premières sont les modules gelés de `crates/kvu/src/heritage/` ; la cinquième est le noyau HIP de K1, lu comme texte par
les tests `su2_noyaux_sources` (jamais compilé ici). Le sha256 de chaque copie est celui du fichier d'origine à `0a7a2de`
(`git -C /chemin/vers/projet/kevinotron show 0a7a2de:<origine> | sha256sum`), gate WP1-AUTONOME.

```
872ebf1008497fc1564778108e205889bc593386429a67f70466a9573eec64fd  crates/kvu/src/heritage/ffi.rs        (origine : src/hip/ffi.rs ; dlopen HIP, M1)
49f79b56daa8b304db2dc70715bf06c75b5ccfc01109651270c8dec44dfd6c6b  crates/kvu/src/heritage/wrap.rs       (origine : src/hip/wrap.rs ; RAII HIP, M1)
36731521a7e0339e202fb061ab4fddf34f21f3cf32b566d8bdaf46b5a0d4d63b  crates/kvu/src/heritage/su2_hb.rs     (origine : src/su2_hb.rs ; référence CPU FP64, K1)
667cb10ea878e25f9a1db9877bdb0db7b2ea2567bd7f870d6c93c65bdf64049d  crates/kvu/src/heritage/manifest.rs   (origine : src/manifest.rs ; provenance K8, avec la recette Unifiee de T8, désormais inutilisée par build.rs)
d04a876a8b5c5e493eb3b5d96eb17685d25740c19cfa395ba47c5731115fe2b2  src/hip/su2_sweep.hip                 (origine : src/hip/su2_sweep.hip ; port K1 de su2_hip.cpp)
```

Recette de provenance du dépôt : `kvu-source/1` (voir `crates/kvu/src/stockage/recette.rs` et `crates/kvu/validation/historique/source_hash_kvu.py`).

## Données de référence lues EN PLACE (trop grosses pour le dépôt ; jamais modifiées, jamais copiées)

| chemin (acemagic, lecture seule) | contenu | sert à |
|---|---|---|
| /chemin/vers/projet/chantiers/vortex_tests_20260927/gpu_port/ref_amd/p2/{L6,L8} | su2_hip de production, 20 pas, dump à chaque pas | G-BIT-HIP-PROD (compat fast-honor-pragmas) |
| /chemin/vers/projet/chantiers/vortex_tests_20260927/gpu_port/cuda/p2/{L6,L8} | su2_cuda, mêmes graines | G-BIT-CUDA (CUBIN sm_120, fmad) |
| (les deux arbres p2 : sha256 de la liste triée des sha256 = bf2d9138…) | | |
| /chemin/vers/projet/chantiers/vortex_tests_20260927/gpu_port/{ref_amd,cuda}/ens | 26 + 20 ensembles L = 6, 8 | G-H09..G-H11 (P3-P5) |
| /chemin/vers/projet/outils/ym4_noyaux/noyau2/data/L8 (65 fichiers, 58 Mo) | chaînes de production C-b de référence (10 V × β ∈ {2,3 ; 16}) | EQC-2 (loi du mode rapide) |
| /chemin/vers/projet/outils/ym4_noyaux/noyau2/data/L{12,16,24}_b2.3.npy | configurations de banc | bancs cond/fibre |
| /chemin/vers/projet/chantiers/wilson_gpu_grandL_20260926/validation_plaquette.json | références de plaquette (V1, AT21) | LOI-2, G-H09 |

## Références de QW1 pour les boîtes de Dirichlet (lot WP-DIR, 25/10/2026)

Copies À L'OCTET du code et des constantes du test QW1 (`chantiers/vision_20261025/calculs/QW1_filtre_dobrushin/`, lecture seule à l'origine) : `box_su2.py` et
`gamma.py` (mesures et méthode Γ de référence, gate G-DIR-OBS et analyses de GATES-DIR.md) ; `qw1_reference.json` est dérivé de `logs/analyse.json`
(sha256 `fc7893a88bbf201f…`) par le script du lot (valeurs de GATES-DIR.md §2, pleine précision).

```
ffc24bb72f4472f2a0df1833436d24b7a27cc1bab23046338f94897eb351fd4c  crates/kvu/reference/qw1/box_su2.py     (origine : QW1/box_su2.py ; heat-bath KP + boîte de Dirichlet en numpy)
439ed2f3d4dd93e542eee21a3cef2243036dfd3625ebf2759d2834e92bb78cae  crates/kvu/reference/qw1/gamma.py       (origine : QW1/gamma.py ; méthode Γ de Wolff, S = 1,5)
8225c69c3fc751d8ae7e39012ccadd3b7bdf5f1dc000b9c7b7b32e0b2c8bcb4e  crates/kvu/reference/qw1/qw1_reference.json (dérivé de QW1/logs/analyse.json ; constantes de GATES-DIR.md §2)
```
