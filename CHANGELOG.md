# Journal des versions

Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) ; versions selon [SemVer](https://semver.org/lang/fr/).
Tant que la version majeure est 0, l'interface (options, formats de manifeste, variables d'environnement) peut changer
d'une version mineure à l'autre ; chaque changement est noté ici.

## [0.5.1] : novembre 2026 (recette du 23 novembre 2026 ; la date de publication est celle du dépôt), première version publique de la ligne 0.5

0.5.0 : version interne, non publiée ; 0.5.1 : première version publique de la ligne 0.5 (au sens du logiciel : ce n'est pas une priorité scientifique) = 0.5.0 interne + reports de correctifs de reprise de la 0.1.1 (PUB-14 standard et `--bord`, PUB-16, `--mesures ""`) + durabilité avant la clôture et gardes d'entrée + corrections de style de clippy.
Les numéros 0.2 à 0.4 sont aussi des versions internes, non publiées séparément ; leurs changements sont intégrés à la 0.5.1. Les fonctionnalités conservées des versions internes sont résumées ci-dessous. Les résultats et limites historiques sont propres aux versions indiquées et ne valent pas validation de la 0.5.1. Les noms de branches, d'agents et de rapports cités dans les sections internes (0.2 à 0.5.0) sont ceux d'un dépôt de travail non public ; ils ne se résolvent pas pour un lecteur extérieur. Les affirmations d'identité au bit ou d'accord de ces sections internes sont des résultats de la qualification interne, non rejoués depuis l'archive publique.
Le code a été écrit par des agents de codage IA sous la direction de l'auteur. Il n'a pas fait l'objet d'une relecture humaine ligne à ligne.


### Mauvaises nouvelles (à lire d'abord)

- État de la validation sur carte et ventilation des 212 tests ignorés : voir la section « État de la validation sur carte (0.5.1) » du `README.md`, bilan détaillé de référence.
- Rejeu des 181 tests de carte (24 novembre 2026, depuis l'archive) : plusieurs ne valent pas preuve (tests qui exigent des références de l'auteur, tests qui s'écartent d'eux-mêmes, délais dépassés, gate OBS-4-L16 déclassée, mémoire de carte occupée) ; chiffres et liste : section « État de la validation sur carte (0.5.1) » du `README.md`.
- Les sorties AMD et NVIDIA ne sont pas identiques au bit. Aucun accord statistique AMD–NVIDIA n'est établi ici pour la 0.5.1.
- Le renommage des noms hérités (`KEVINOTRON_*`, profils `acemagic` et `gamer`), annoncé pour la 0.2, n'est pas fait ; il est reporté à la 0.6 (avec des alias pour les anciens noms).
- La plupart des fichiers source n'ont pas d'en-tête SPDX (la licence Apache-2.0 est portée par `LICENSE`, `NOTICE` et le champ `license` de `Cargo.toml`) ; l'ajout des en-têtes est prévu pour la 0.6 (il change l'empreinte des sources).
- Ce qui entre dans le public est bien plus volumineux que la 0.1.1 (modules [EXPÉRIMENTAL] écrits par des agents). Le code a été écrit par des agents de codage IA sous la direction de l'auteur. Il n'a pas fait l'objet d'une relecture humaine ligne à ligne.
- Limites propres à chaque module : `docs/ETAT_v0.5.md`, `docs/ETAT_v0.4.md`, `docs/ETAT_v0.3.md` (rubriques « Mauvaises nouvelles »).

### Corrigé par rapport à la 0.5.0 interne

- **PUB-14** (arrêt brutal pendant la thermalisation : `measurements.csv` restait vide et `--reprendre` refusait le run, code 4) et **PUB-16** (le `COMPLETE` d'un run repris
  portait une phrase fixe : `kvu store verifier` le déclarait rouge, code 8) : correctifs de la 0.1.1 reportés (trois commits ; fichier `crates/kvu/src/su2/run.rs`).
  Tests ajoutés : `pub14_kill9_en_thermalisation_reprise_acceptee`, `pub16_run_repris_passe_store_verifier`, `pub16_run_repris_avec_mesures_passe_store_verifier`
  (`crates/kvu/tests/su2_reprise_mesures.rs`). Ces deux défauts avaient été reproduits sur le binaire de la 0.5.0 interne ; ces tests passent sur la 0.5.1 (dans les cas qu'ils couvrent : `kill -9` pendant la thermalisation, chemin standard ; run repris vérifié par `kvu store verifier`).
- **Audit des correctifs de la 0.1.1** (un par un, contre cette version) : PUB-01, PUB-02, PUB-04, PUB-09 et PUB-13 y sont présents (sous la forme de la ligne de développement pour PUB-01 et PUB-13), avec des tests ;
  deux comportements de la 0.1.1 manquaient encore, reportés avec leurs tests : **PUB-14 pour `--bord`** (en-têtes de `therm_blocs.csv` et `blocs.csv` écrits et synchronisés dès la création ; test `pub14_bord_kill9_en_thermalisation_reprise_acceptee`,
  `su2/bord.rs`) et **`--mesures ""` vaut « sans mesures » à la reprise** (test `mesures_vide_vaut_sans_mesures`, `su2/run.rs`). Les tests `pub01_*`, `pub02_bord_et_mesures_refuse`, `run_sans_mesures_sans_compagnon` de la 0.1.1 sont rétablis dans `su2_reprise_mesures.rs`.
  PUB-05 est traité en partie (voir `docs/DEFAUTS_CONNUS.md`) ; les autres défauts de la 0.1.1 sont des documentations (même fichier).
- **Durabilité avant la clôture** (chemin standard de `su2 gen`, après relecture de code par un modèle d'une autre famille ; trois situations distinctes) : (a) *clôture* : les fichiers consignés au manifeste sont synchronisés avant la publication de `manifest.json` puis de `COMPLETE` ; (b) *reprise après arrêt du processus* (`kill -9`) : protocole de `reprise.champs` / `reprise.json` testé à vrai `kill -9` (tests `su2_reprise_mesures`, `su2_cpu`) ; une génération désignée par `reprise.json` n'est plus écrasée par l'écriture suivante après deux arrêts successifs (test `deux_arrets_successifs_ne_perdent_pas_la_generation_designee`) ; (c) *reprise après coupure de courant* : NON établie et non testée (PUB-05 n'est pas résolu : voir `docs/DEFAUTS_CONNUS.md`). L'écriture atomique rend désormais l'erreur de synchronisation du répertoire (test `ecrire_atomique_propage_l_echec_de_synchronisation_du_repertoire`).
- **Gardes d'entrée** (relecture de code par un modèle d'une autre famille ; changements de comportement sur entrées extrêmes, sans effet sur les entrées valides) : `su2 thermo mesure` : produits et volumes contrôlés (`checked_mul`, tests `entrees_extremes`) ; `etat.bin` de `su2 spectro` : nombre de liens converti par `usize::try_from` et identifiant de balayage refusé au-delà de `u32::MAX` ; `lire_etat` de la reprise essaie les deux fichiers de champs ; garde `b >= usize::BITS` avant un décalage dans `gpu_torelon.rs`. Aucune boucle numérique n'est modifiée.
- `cargo clippy --locked -p kvu --all-targets -- -D warnings` : de 65 lints à zéro (avec et sans la fonctionnalité `cuda` ; journaux `clippy_051b.log` et `clippy_cuda_051b.log` du paquet de publication, non joints à l'archive). Corrections de style ; l'équivalence numérique n'est pas déduite du mot « style » : voir les sept essais ci-dessous (`is_multiple_of`, `div_ceil`,
  `slice::from_ref`, un `0 *` inutile, deux branches identiques fusionnées) ; pour les boucles indexées, les types complexes et les fonctions à beaucoup d'arguments,
  un `#[allow(clippy::…)]` local et commenté (aucune boucle numérique réécrite ; les essais processeur ne montrent aucune différence) ; pour les listes de documentation, un `#![allow(...)]` de fichier.
- Trois mentions d'un projet sans rapport retirées de commentaires et de fichiers d'exemple.
- Sur sept essais processeur, les fichiers et champs effectivement comparés sont identiques au bit entre la 0.5.0 interne (e9f9f12) et la 0.5.1 ; la liste exhaustive des essais, des éléments comparés et des exclus (manifestes, `COMPLETE`, journaux, champs de provenance de `archive.json` et `resume.json`, etc.) est dans `docs/PUBLICATION.md` §5. Aucune garantie générale n'en est tirée.
- Numéro de version 0.5.0 → 0.5.1 (`kvu --version`, `Cargo.toml`, `Cargo.lock`, `CITATION.cff`, `.zenodo.json`, documents) ; `CITATION.cff` portait encore la version et le DOI de la 0.1.0 : corrigé.
- Liste de référence des noms de tests : `scripts/recette/noms_tests/v0.5.1.cpu.txt` (2 073 tests listés, 1 861 actifs, 212 ignorés ; `v0.5.0.cpu.txt` conservée).

### Recette de cette version

Mesurée le 23 novembre 2026 sur un PC Linux x86_64 (AMD Ryzen AI Max+ 395, Ubuntu 26.04 ; cargo et rustc 1.97.0, Python 3.14.4, zstd 1.5.7, numpy et scipy installés), par `scripts/recette_locale.sh` sur l'archive `kvu-0.5.1.zip` de l'étiquette `v0.5.1` (sans `.git` ; son SHA-256 est dans `SHA256SUMS`, la fiche de publication et la description du dépôt, et ne peut pas figurer dans l'archive elle-même), dépendances téléchargées depuis crates.io avec `--locked` (`CARGO_HOME` vide), `HOME` vide, `KEVINOTRON_SSD_MIN_GO=0`, aucune carte graphique utilisée, 4 tâches de compilation et 4 fils (certains tests lancent eux-mêmes `kvu`, dont jusqu'à 8 fils) :

| étape | résultat |
|---|---|
| `cargo build --release --locked -p kvu` | rc 0 |
| `cargo test --release --locked -p kvu` | **2073 tests listés dans 116 exécutables : 1861 actifs, tous passés, 0 échec ; 212 ignorés** ; les résumés de cargo additionnent 1 866 « passed », soit 5 de plus que les 1 861 tests actifs listés (provenance de l'écart non établie par un journal : processus enfants de tests, hypothèse) |
| noms des tests | aucun test de `v0.1.0.cpu.txt` ni de `v0.5.0.cpu.txt` retiré ni désactivé ; chaque test actif exécuté ; référence `v0.5.1.cpu.txt` |
| `cargo clippy --locked -p kvu --all-targets -- -D warnings` | **VERT**, avec et sans `--features cuda` |
| `cargo test --release --locked -p kvu --features cuda --no-run` | rc 0 : compilation seulement, aucune exécution sur carte |

212 tests ignorés, dont 181 tests de carte, ne sont pas rejoués par cette recette ; les tests de carte ont été rejoués à part, depuis l'archive, sur AMD et NVIDIA : résultats dans la section « État de la validation sur carte (0.5.1) » du `README.md`.

## [0.5.0] : novembre 2026 (version interne, non publiée), vue d'ensemble depuis la 0.1.1

Cette version est bâtie sur la ligne de développement (`integration/v0.5`) et non sur la 0.1.1 : elle contient les versions internes 0.2 à
0.4, qui n'ont pas été publiées séparément. Pour la préparation de cet instantané interne, aucun changement fonctionnel n'a été fait : elle ne touchait que
la documentation, les métadonnées et le retrait de fichiers (voir « Préparation de la publication » ci-dessous) ; la 0.5.1, elle, corrige des défauts de reprise (section précédente). Les options
ajoutées sont expérimentales et éteintes par défaut.

### Mauvaises nouvelles (à lire d'abord)

- **(Corrigé dans la 0.5.1, voir plus haut.) PUB-14 et PUB-16, corrigés dans la 0.1.1, sont de nouveau présents dans la 0.5.0 interne** : cette version ne reporte pas ces deux correctifs
  (commits internes de la branche `integ/c15-arret-brutal`, non fusionnés). Un `kill -9` pendant la thermalisation laisse
  `measurements.csv` vide et `--reprendre` refuse le run (code 4) ; le `COMPLETE` d'un run repris ne porte pas l'empreinte du manifeste
  et `kvu store verifier` le déclare rouge (code 8). Parades dans `crates/kvu/GUIDE.md` et `docs/DEFAUTS_CONNUS.md`.
- (Corrigé dans la 0.5.1.) `cargo clippy -D warnings` est rouge sur la 0.5.0 interne (65 lints de style dans les modules ajoutés depuis la 0.1.1 ; la 0.1.1 était verte) : `scripts/recette_locale.sh` rend `VERDICT ROUGE` ; voir « Recette de cette version ».
- Les tests de carte graphique (`#[ignore]`) ne sont pas rejoués par la recette locale ; pour la 0.5.0 interne, ils n'ont pas été rejoués depuis l'archive publique (état de la 0.5.1 : section « État de la validation sur carte (0.5.1) » du `README.md`).
- Les sorties AMD et NVIDIA ne sont pas identiques au bit ; aucun accord statistique AMD–NVIDIA n'est établi ici.
- Le renommage des noms hérités (`KEVINOTRON_*`, profils `acemagic` et `gamer`), annoncé pour la 0.2, n'est pas fait ; il est reporté à la 0.6 (avec des alias pour les anciens noms).
- Limites propres à chaque module : `docs/ETAT_v0.5.md`, `docs/ETAT_v0.4.md`, `docs/ETAT_v0.3.md` (rubriques « Mauvaises nouvelles »).

### Préparation de la publication (état de la 0.5.0 interne ; la 0.5.1 y ajoute les correctifs ci-dessus)

- Retirés de l'instantané : `rapports/`, la campagne interne WP9-AV1 et ses pré-enregistrements, les journaux et résultats de validation interne
  (`crates/kvu/validation/**/resultats*`), comme pour la 0.1 ; en plus, huit scripts de campagne à chemins absolus de la machine de l'auteur
  (`scripts/accord_xi.*`, `balayage_xi2.sh`, `analyse_point_xi.py`, `test_aniso_*.sh`, `test_ops_bruts.sh`, `verif_ops_bruts.py`).
- Neutralisés : chemins de la machine de l'auteur dans `scripts/lecture_graine/`, `docs/ETAT_v0.3.md`, et un fichier de test (`tests/donnees/reprise_0_1_1/manifest.json`).
- Remis dans l'état publié de la 0.1.1 : `docs/NOTE_0.1.1.md`, `scripts/verifier_mesures_en_ligne.py`, `docs/DEFAUTS_CONNUS.md` (avec l'avertissement ci-dessus),
  `crates/kvu/validation/machine/test_build_auto*.sh` (chemins d'essai neutres).
- README, CHANGELOG, `CITATION.cff`, `.zenodo.json`, `AI_USE.md`, `LICENCES_TIERCES.md` mis à jour.

### Ajouté depuis la 0.1.1 (toutes options éteintes par défaut)

- `su2 spectro` (`run`, `neutre`, `gpu`, `gpu-torelon`) : spectroscopie de torelons et de tubes de flux sur boîte rectangulaire (CPU, AMD, NVIDIA) ;
  options `--impulsions`, `--impulsions-vecteurs`, `--blocage`, `--deux-niveaux`, `--profil`, `--secteur-flux`, `--xi0`.
- `su2 thermo` (`mesure`, `analyse`) : méthode intégrale sur boîtes rectangulaires, carte graphique obligatoire pour `mesure` ; `--series`, `--torsion-fixe`,
  `--decalage` (couture décalée sur carte), `--beta-s` (couplages séparés), `--lien-integre` (estimateur multi-hit par intégration analytique de la moyenne conditionnelle d'un lien, dans les conditions indiquées ; G. Parisi, R. Petronzio, F. Rapuano, *Phys. Lett. B* 128 (1983) 418 ; aucune promesse d'exactitude du résultat Monte-Carlo).
- `su2 tube`, `su2 gen --cotes`, `su2 gen --decalage` (tores obliques, CPU), `su2 gen --torsion` sur carte ; `obs champs` ; `rg correlateur`, `rg apparier` ;
  mesure en ligne `corr` ; `rg action --erreurs bootstrap`.
- `intrication` (entropie de Rényi-2 par répliques, CPU F64) ; `experience` (carnet d'expérience préenregistré et vérifiable).
- Scripts Python de lecture et d'analyse : `scripts/thermo/`, `scripts/spectro/`, `scripts/tube/`, `scripts/lecture_graine/`, `scripts/thermo/transport*.py`.

### Recette de la 0.5.0 interne (historique)

Mesurée le 23 novembre 2026 sur un PC Linux x86_64 (AMD Ryzen AI Max+ 395, Ubuntu 26.04 ; cargo et rustc 1.97.0, Python 3.14.4, zstd 1.5.7, numpy et scipy
installés), par `scripts/recette_locale.sh` sur un `git archive` de la version (sans `.git`), dépendances téléchargées depuis crates.io avec `--locked`
(`CARGO_HOME` vide), `HOME` vide, `KEVINOTRON_SSD_MIN_GO=0`, aucune carte graphique utilisée, 4 tâches de compilation et 4 fils (certains tests lancent
eux-mêmes `kvu` avec jusqu'à 8 fils) :

| étape | résultat |
|---|---|
| `cargo build --release --locked -p kvu` | rc 0 |
| `cargo test --release --locked -p kvu` | **2056 tests listés dans 115 exécutables : 1844 actifs, tous passés, 0 échec ; 212 ignorés** ; 1849 « passed » additionnés sur 123 lignes (5 lignes d'enfants) |
| noms des tests | aucun test de `v0.1.0.cpu.txt` retiré ni désactivé ; chaque test actif exécuté ; référence `v0.5.0.cpu.txt` écrite |
| `cargo clippy --locked -p kvu --all-targets -- -D warnings` | **ROUGE** : 65 lints de style (boucles indexées, types complexes, documentation) dans `src/su2/spectro/`, `src/su2/tube/` et quelques autres fichiers des modules ajoutés depuis la 0.1.1 ; la 0.1.1 était verte. Non corrigé (aucun changement de code pour la publication). `scripts/recette_locale.sh` rend donc `VERDICT ROUGE` à cette étape ; `--sans-clippy` l'écarte |
| compilation `--features cuda --no-run` | rc 0 (134 s) |

Historique de la 0.5.0 interne : 212 tests ignorés (total ; leur ventilation pour la 0.5.0 n'est pas établie ici), et aucun rejeu de carte par cette recette ; état de la 0.5.1 : section « État de la validation sur carte (0.5.1) » du `README.md`.

## [0.5.0, détail par version interne] : novembre 2026

Numéro de version du paquet `kvu` : 0.4.0 devient 0.5.0 (`crates/kvu/Cargo.toml`, `Cargo.lock` ; `kvu --version` affiche `kvu 0.5.0`). Branche `integration/v0.5` = `integration/v0.4` plus les outils ci-dessous
(rapports internes non publiés). Toutes les options sont éteintes par défaut ; les sorties des chemins existants ont été comparées à la version précédente lors de la qualification interne (non rejouée depuis l'archive).

- Couture décalée sur carte (`geometrie/coutures-gpu-1122`) : `su2 thermo mesure --decalage`, `--decalage-b`, `--decalage-axe` (AMD et NVIDIA ; `su2 gen` reste processeur). Refusée avec torsion, anisotropie, `--lien-integre` et `--beta-s`.
- Impulsions hors axe : orbites regroupées en un seul passage par groupes de 32 voies (`codex/orbites-groupees-1122`) ; mêmes fichiers qu'avant (qualification interne) ; les temps mesurés sont dans des rapports internes non publiés.
- Flux anisotrope et secteur (1,1,1) : l'opérateur du secteur (1,1,1) de `su2 spectro gpu-torelon --secteur-flux` existe (il était refusé) (`codex/flux-anisotrope-111-1122`) ; le secteur (1,1,0) est inchangé.
- `su2 thermo mesure --beta-s BS` : couplage des plaquettes d'espace fixé, `--betas` devient la liste des couplages temporels ; colonnes `beta_s,st,ss`, séries `_st.bin` et `_ss.bin`, bloc `couplages_separes` du manifeste.
- `su2 thermo mesure --lien-integre` : estimateur à lien intégré (multi-hit par intégration analytique de la moyenne conditionnelle d'un lien ; aucune promesse d'exactitude du résultat Monte-Carlo) : colonnes et séries `y0_t`, `ys_t`, `zs_s`, `y0_tot`, `ys_tot` ; noyau `su2_lien.cu` ; refusé avec torsion, torsion fixe et couture.
- Lecteur des séries de torsion corrigé (`scripts/.../lire_series.py` : bootstrap conjoint si le manifeste annonce l'appariement, dimensions contradictoires refusées, poids non finis refusés) ; garde G4 : un lecteur absent est un échec.
- `scripts/thermo/transport.py`, `transport_mesures.py` (Codex) : générateur de campagne et lecteur de transport à longueur physique commune (Python standard seulement ; voir `scripts/thermo/TRANSPORT.md`).
- Tests : voir `docs/ETAT_v0.5.md` (nombres de la qualification). `KEVINOTRON_SSD_MIN_GO=0` reste nécessaire quand `/tmp` a moins de 100 Go libres.

## [0.4.0] : 22/11/2026 (version interne, non publiée séparément ; ses changements sont intégrés à la 0.5.1)

Numéro de version du paquet `kvu` : 0.1.0 devient 0.4.0 (`crates/kvu/Cargo.toml`, `Cargo.lock` ; `kvu --version` affiche `kvu 0.4.0`). Les fichiers qui désignent
l'archive publiée 0.1.0 (`CITATION.cff`, README, recette locale) ne changent pas. Branche `integration/v0.4` = `integration/v0.3` + quatre outils (fusion n°3,
rapport interne non publié). Toutes les options ci-dessous sont éteintes par défaut ; les chemins existants ont été comparés à `integration/v0.3` lors de la qualification interne (processeur, AMD et NVIDIA ; non rejoué depuis l'archive publique).

- `su2 gen --decalage v0,v1,v2,v3` (avec `--decalage-axe`) : boîtes de glu sur un tore oblique (couture décalée). Processeur seulement.
- `su2 spectro gpu|neutre --impulsions-vecteurs` : impulsions hors axe (orbites cubiques, par exemple 2:2:1). Processeur, AMD et NVIDIA ; identique au bit au binaire de sa branche.
- `su2 spectro gpu-torelon --secteur-flux` (avec `--flux-sans-corr`) : tube de flux dans le secteur (1,1,0). AMD et NVIDIA ; l'opérateur du secteur (1,1,1) est refusé.
- `su2 thermo mesure --series` et `--torsion-fixe` : séries des rapports de torsion et pile fixe. AMD et NVIDIA ; pas de voie processeur.
- Qualification : 1 833 tests processeur réussis, 0 échec, 197 ignorés (tests de carte). `intrication_cli` exige `KEVINOTRON_SSD_MIN_GO=0` quand le disque a moins de 100 Go libres.
  Essais de fumée : 33 OK et 1 KO (`rg gates-gpu`, connu) sur AMD et sur NVIDIA ; processeur 27 OK et 7 refus attendus. Carte et limites : `docs/ETAT_v0.4.md`.

## [0.3.0] : 22/11/2026 (intégration interne v0.3, non publiée séparément ; ses changements sont intégrés à la 0.5.1)

- Branche `integration/v0.3` : `spectro/integ-1121` (contient `integration/v0.2`, `integ/1119`, `kvu/impairs-1120`) plus les scripts
  `accord_xi`, `balayage_xi2`, `analyse_point_xi` de `spectro/aniso-1121`. Aucun changement de code Rust.
- Qualification : 1 798 tests processeur verts (`KEVINOTRON_SSD_MIN_GO=10` requis pour 16 tests écrivant sous `/tmp`), 141 tests GPU
  ignorés verts sur HIP, identité au bit CUDA contre les binaires de production (critère pair, critère impair, `spectro gpu`,
  `gpu-torelon`). Carte et limites : `docs/ETAT_v0.3.md`.

### Fusion du 22/11/2026 (agent `fusion`, sur `integration/v0.3`)

- `spectro/xi0-cpu-1122` (773d194) : `su2 spectro run|neutre --xi0 ξ₀` (réseau anisotrope sur processeur ; `plaq_ss_st.csv`, clé `xi0` du
  manifeste) ; éteinte par défaut, sorties identiques au bit sans l'option ou à ξ₀ = 1.
- `spectro/profil-tube-1122` (e24012d) : `su2 spectro gpu-torelon --profil τ1,τ2,… [--profil-paquet N --profil-niveau K]` (profil du tube de
  flux enroulé : noyau `k_rho`, cartes Φ/ρ, `profil.bin`) ; éteinte par défaut, inerte sans l'option. Aucun conflit textuel ; un conflit
  sémantique corrigé : `ParamTorelon` a deux champs de plus (`profil`, `profil_niveau`), `tests/su2_spectro_gpu.rs` initialisait la structure sans eux.
- `scripts/spectro/analyse_spectro.py`, `e_eff` : le test `python3 scripts/spectro/test_e_eff.py` couvre les périodes temporelles T = 16, 32, 48 et 64 pour des énergies E·a = 0,05, 0,3 et 0,9 (forme cosh) : il retrouve l'énergie connue à 10⁻⁶ et reproduit le défaut corrigé dans ces cas (l'ancienne garde `min(e·a, 700)` donnait NaN pour T > 32) ; pour T ≤ 32, les résultats du test sont identiques au bit à l'ancienne garde. Aucune généralisation à d'autres périodes ou arguments.

### Fusion n°2 du 22/11/2026 (agent `fusion-2`, sur `integration/v0.3`)

- `spectro/deux-niveaux-1122` (803143d) : `su2 spectro gpu|gpu-torelon --deux-niveaux D,N1[,S1]` (algorithme à deux niveaux : masque de liens gelés dans
  les noyaux F32 et F64, cumul sur la carte des opérateurs, impulsions et blocage ; `--dn-hote` = ancienne voie hôte ; `deux_niveaux.bin`,
  `corr_deux_niveaux.csv`) ; éteinte par défaut, sorties identiques au bit sans l'option. Conflits résolus : `spectro/cli.rs` (fichiers vidés en fin
  de boucle : `fprof` et `fcd`), `spectro/gpu_torelon.rs` (`calc_phi` rend la carte Φ du profil, reprise par `mesurer`).
- `scanner/v03-1122` (7da02aa) : `obs champs` pour des côtés quelconques ; `su2 gen --cotes` sur carte (`--backend hip|cuda`, F64) ; `su2 gen --cotes … --xi0 X`
  (`plaq_ss_st.csv`). Éteintes par défaut. Aucun conflit.

## [0.1.1] — 2026-11 (correctifs de reprise et de documentation)

Aucun changement de numérique : les sorties d'un run sans `--mesures`, `measurements.csv`, `therm.csv` et les dumps sont
identiques à la 0.1.0. Détail des défauts : `docs/DEFAUTS_CONNUS.md` §3 ; vérification de ses anciens runs : `docs/NOTE_0.1.1.md`.

### Corrigé

- **`su2 gen --reprendre` et les mesures en ligne** (PUB-01, PUB-13) : la spécification `--mesures` est enregistrée à la création
  du run dans `mesures_en_ligne.json` (fichier NOUVEAU, pour les runs lancés avec `--mesures`) et rechargée à la reprise ; une
  `--mesures` différente est refusée (code 2) ; les fichiers des mesures en ligne sont ramenés au point de reprise (plus de lignes
  en double après un arrêt brutal). Avant toute troncature, le préfixe CONSERVÉ de chaque fichier (les lignes jusqu'au point de
  reprise) est validé : en-tête, colonnes, calendrier des pas, clés et nombre de lignes par pas ; un préfixe malformé, incomplet
  ou déjà doublonné fait refuser la reprise (code 2) sans modifier le fichier ; sinon le fichier est ramené au point de reprise. Un run créé par la 0.1.0 avec des mesures n'a pas `mesures_en_ligne.json` : il faut repasser la même `--mesures`.
- **Arrêt brutal du processus pendant la thermalisation** (PUB-14) : les en-têtes de `therm.csv` et `measurements.csv` (avec
  `--bord` : `therm_blocs.csv` et `blocs.csv`) sont écrits et synchronisés (fichiers, puis répertoire) dès leur création ; après
  un `kill -9` en thermalisation, la 0.1.0 laissait un CSV vide et refusait la reprise (code 4). La reprise repart du dernier point
  de reprise valide ; avant le premier point de reprise, il n'y a rien à reprendre ; une coupure de courant relève de PUB-05.
  Aucun octet de sortie ne change. Tests à vrai `kill -9` (`tests/su2_reprise_mesures.rs`,
  `pub14_kill9_en_thermalisation_reprise_acceptee`, `pub14_bord_kill9_en_thermalisation_reprise_acceptee`).
- **`COMPLETE` d'un run repris** (PUB-16) : `rc=0` et `manifest_sha256=<sha256 du manifeste>`, comme un run neuf, au lieu d'une
  phrase fixe ; un run repris puis terminé passe `kvu store verifier` (la 0.1.0 le déclarait rouge, code 8).
- **`scripts/verifier_mesures_en_ligne.py`** (NOUVEAU, lecture seule) : contrôle les fichiers de mesure d'anciens runs
  (`docs/NOTE_0.1.1.md`).
- **`su2 gen --bord --mesures …`** (PUB-02) : refusé (code 2) ; la 0.1.0 clôturait en COMPLETE sans mesure.
- **`scripts/build_auto.sh`** (PUB-04) : la garde de la cible de production ne dépend plus du préfixe.
- **Tests machine** (PUB-09) : `KVU_TEST_MACHINE_ATTENDUE=acemagic|gamer` ; `docs/MACHINES.md` précise l'ordre de recherche du `kvu.toml`.

### Changements de comportement

- Refus avec le code 2 là où la 0.1.0 clôturait en silence (`--bord` avec `--mesures` ; reprise avec une spécification différente).
- Nouveau fichier `mesures_en_ligne.json` dans le répertoire d'un run avec `--mesures`.
- `COMPLETE` d'un run repris : `rc=0` + `manifest_sha256=…` (PUB-16).

### Documenté, non corrigé

PUB-03, 05, 06, 07, 08, 10, 11, 12, 15, 18, 19 et la variable d'essai `KVU_INJECTION_FAUTE_GPU` : `docs/DEFAUTS_CONNUS.md` §3.
PUB-18 et PUB-19 sont deux limites aux marges de la chaîne de Markov ; elles sont toujours présentes dans la 0.5.1 (compteur de balayages sur 32 bits ; tirage de Creutz inchangé) : au-delà de 2³² balayages de bain thermique dans un
même run, pour une même chaîne (le compteur 32 bits du générateur aléatoire repasse à zéro, sans refus ni avertissement ; définition du compte et de la borne : PUB-18) ; β extrêmement petit en F64 (k ≤ 6 est la norme de la somme des agrafes d'un lien ; à β·k sous environ 2,8·10⁻¹⁷ aucun tirage n'est accepté : run très lent, échec à la FIN avec le code 8 ; paramètre à éviter : β < 10⁻³). Bornes et comportement observable : `docs/DEFAUTS_CONNUS.md`, PUB-18 et PUB-19.
PUB-15 : `kvu regen` ne rejoue pas un run `su2 gen` (ni son `regen_bloc3.json`) ; le README et le GUIDE ne le promettent plus
et donnent la parade (relancer l'`argv` du manifeste, comparer les sha256).

## [0.1.0] — 2026-11 (première version publique)

Premier instantané public du dépôt de travail (`main` 9362d77 du 2 novembre 2026, plus les changements de publication
ci-dessous).

### Contenu

- Générateur SU(2) Wilson 4D `su2 gen` : bain thermique + surrelaxation, Philox4x32-10 à compteur, F64 de référence et F32, source unique de noyaux pour HIP et CUDA, référence CPU F64, lots de chaînes, reprise au bit, mesures en ligne,
  bord gelé ; boîtes de Dirichlet `su2 boite`.
- Bain conditionnel (`cond`), fibres (`fibre`), observables (`obs` : Wilson, Polyakov, flot, topologie), blocage RG (`rg` :
  Bałaban L_b = 3, Swendsen b = 2, flot de couplages, blocage quasi-local), hessienne (`hess`), registre de groupes et
  générateur Sp(4) (`groupe`, `gen`), store, rapatriement, régénération et file de jobs (`store`, `fetch`, `regen`, `queue`),
  validation et bancs (`valide`, `bench`), détection de machine (`doctor`, `etat`).
- Modules M1 à M4 (instruments) : archive vérifiable `kvu-archive/1` ; couplages d'une action effective tronquée
  (`rg action`) ; flux de bloc (`rg flux`) ; torsion de 't Hooft (`su2 gen --torsion`, CPU F64).

### Corrigé (préparation de la publication)

- **Compression zstd sous plafond de tâches** : `zstd -T4` échouait (code 104, « Failed creating I/O thread pool ») sous
  `TasksMax = 8`. Compression, test et lecture passent maintenant `--single-thread` et, si le zstd installé la connaît
  (≥ 1.5.4), `--no-asyncio` : les options sont sondées une fois par processus, pour qu'un zstd plus ancien continue de
  fonctionner. Correctif d'origine : branche interne `sol/fix-zstd-single-thread-20261104` (4 novembre 2026).
- **`rg action` lit la sortie de `su2 gen`** : il ne lisait que la clé `beta` des manifestes de production et rendait 4 sur
  un ensemble produit par `kvu su2 gen` (clé `betas`). Une chaîne unique est maintenant lue ; avec une seule chaîne, employer
  `--jackknife blocs:B`.
- **`HOME` absent** : repli sur le dossier courant au lieu d'un dossier personnel codé en dur.
- **Tests hermétiques** : le dossier de travail des tests de stockage est sous `target/` (`CARGO_TARGET_TMPDIR`) ; les tests
  des gates historiques ont leur propre store et ne dépendent plus de la place libre de la machine (`KVU_SSD_MIN_GO`, variable des scripts de validation, pas un alias lu par le binaire) ;
  aucun test ne crée de dossier hors du dépôt et du dossier temporaire.

### Changé

- Le binaire s'annonce `kvu` (aide, `Usage:`, préfixe des messages d'erreur `kvu <sous-commande> : …`) au lieu de
  `kevinotron`, nom du moteur interne dont il est issu. Les variables d'environnement gardent le préfixe `KEVINOTRON_`.
- `kvu.toml` à la racine est un **modèle** neutre (profils `inconnue` et `location`), qui ne reconnaît aucune machine. Les
  profils des deux machines de référence sont la fixture de test `crates/kvu/kvu.toml`.
- Guide d'utilisation : `crates/kvu/GUIDE.md` (anciennement `README_UNIFIE.md`), réécrit pour le dépôt autonome.

### Ajouté

- `scripts/recette_locale.sh` (recette sur une machine), `scripts/recette/noms_tests.py` et la référence
  `scripts/recette/noms_tests/v0.1.0.cpu.txt` (liste des tests selon cargo), `scripts/exemples_readme.sh` (exécute les
  exemples du README).
- `README.md`, `CHANGELOG.md`, `CITATION.cff`, `.zenodo.json`, `AI_USE.md`, `LICENCES_TIERCES.md`, `docs/PUBLICATION.md`.

### Défauts connus

- **OBS-4-L16 déclassée** (échec documenté, non bloquant) : la charge topologique « trèfle » non améliorée de `obs flot` n'est
  pas quantifiée dans ce test historique (L = 16, β = 2,5, t/a² = 3 : 5 configurations sur 20 satisfaisaient le critère de proximité à 0,1 d'un entier, contre 18 requises, sur CUDA comme sur HIP ; rejeu identique au bit
  sur CUDA ; test de la 0.1.0, non rejoué pour la 0.5.1). Aucune cause n'est établie publiquement. Observable marqué
  expérimental ; `obs topo` donne la charge améliorée. Test et seuil inchangés. Voir `docs/DEFAUTS_CONNUS.md`.

### Retiré de l'instantané public

- `rapports/`, la campagne interne WP9-AV1 (`crates/kvu/validation/wp9_av1/`, pré-enregistrements associés) et les journaux
  de validation interne (`crates/kvu/validation/**/resultats*`) : voir `docs/PUBLICATION.md`.

### Historique et nom

- kvu succède à l'outillage interne « kevinotron » de l'auteur ; les dépôts Zenodo exploratoires de mai 2026 qui portent ce nom
  (10.5281/zenodo.20412608, 10.5281/zenodo.20415354) précèdent kvu et n'en sont pas des résultats (voir le README).
- Prévu pour **0.6** (annoncé d'abord pour la 0.2, reporté) : renommage des variables `KEVINOTRON_*`, des défauts `~/kevinotron_store` et `/tmp/kevinotron_gpu.lock`
  et des profils `acemagic`/`gamer`, avec des alias pour les anciens noms.

### Recette de cette version

Mesurée le 8 novembre 2026 sur un PC Linux x86_64 (Intel Core i5-14600KF, 32 Go, Ubuntu 26.04 ; cargo et rustc 1.97.0,
Python 3.14 avec numpy 2.5.3 et scipy 1.18.1, zstd 1.5.7), par `scripts/recette_locale.sh` sur un **clone propre** de la
branche (commit `6edc4e3`), dépendances téléchargées depuis crates.io avec `--locked` (`CARGO_HOME` vide), `HOME` vide, nom
d'hôte neutre, **aucun accès aux GPU** (périphériques interdits par le cgroup), 8 tâches de compilation et 8 fils :

| étape | résultat | durée |
|---|---|---|
| `cargo build --release --locked -p kvu` | rc 0 | 23 s |
| `cargo test --release --locked -p kvu` | **1578 tests compilés dans 82 exécutables : 1463 actifs, tous passés, 0 échec ; 115 ignorés** (tests GPU et tests lourds, `#[ignore]`) | 884 s (compilation des tests comprise) |
| liste des tests selon cargo | chaque test actif listé a été exécuté ; référence `scripts/recette/noms_tests/v0.1.0.cpu.txt` écrite | — |
| `cargo clippy --locked -p kvu --all-targets -- -D warnings` | rc 0 | 10 s |
| `cargo test --features cuda --no-run` | rc 0 (le backend CUDA compile ; rien n'est lancé) | 43 s |
| `scripts/exemples_readme.sh` | les 11 commandes du README rendent 0, sorties attendues présentes | 2 s |
| tests Python de la recette (`scripts/tests/`) | 7/7 | < 1 s |

La somme des lignes « test result » de cargo donne 1468 « passed » : cinq tests de `tests/su2_cpu.rs` relancent leur
propre exécutable dans un processus enfant, dont le résumé s'ajoute. Le décompte exact est celui de la liste (1463 actifs).
Aucun dossier n'a été créé hors de l'arbre de travail et du dossier temporaire. Les tests GPU n'ont pas été lancés pour
cette recette.
