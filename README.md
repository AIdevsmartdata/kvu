# kvu

**kvu** est un générateur et un banc de mesure pour les théories de jauge sur réseau, centré sur la théorie de Yang–Mills SU(2)
à action de Wilson en quatre dimensions. Un seul code de noyaux GPU sert deux backends, HIP (AMD) et CUDA (NVIDIA), compilés à
l'exécution, à côté d'une référence CPU en double précision. Les commandes qui écrivent un run (par exemple `su2 gen`, `su2 boite`) écrivent un manifeste de provenance (binaire, commit,
empreinte des sources, graines, options) ; le commit git est celui du dépôt si le binaire est construit depuis un dépôt git (suffixe `+sale` si l'arbre est modifié) ; depuis une archive sans `.git`, il est absent ou marqué `+non_verifie` (variable `KEV_GIT_HEAD` déclarée à la compilation). Les configurations peuvent être archivées
et vérifiées (`kvu su2 archive verifier` recalcule les empreintes SHA-256 contre l'index de l'archive ; il ne prouve pas l'authenticité de cet index).
Aucune garantie générale de reproductibilité au bit n'est donnée. Ce que des tests de l'archive établissent, sur processeur (backend `cpu`, F64, petits réseaux) :
un run interrompu puis repris donne les mêmes fichiers que le même run non interrompu (dumps, `therm.csv`, `measurements.csv` ; tests `su2_cpu::reprise_1_standard_nouveau_processus`, L = 4, une et deux chaînes, `su2_cpu::reprise_1_bord_nouveau_processus` pour `--bord`,
`su2_torsion::t6_reprise_avec_torsion`, `su2_reprise_mesures::*` pour les mesures en ligne, `spectro_neutre_reprise::reprise_hors_axe_identique_au_bit`), et, sur sept essais processeur, les fichiers comparés de la 0.5.1 sont identiques à ceux de la 0.5.0 interne (`docs/PUBLICATION.md` §5). Rien de plus n'est établi : ni pour d'autres arguments, formats ou environnements, ni entre backends, ni entre machines ;
sur carte, voir [« État de la validation sur carte (0.5.1) »](#état-de-la-validation-sur-carte-051). Trois mots à ne pas confondre : **reprise** (`su2 gen --reprendre` d'un dossier de run
interrompu, au dernier point de reprise), **rejeu** (`kvu regen` d'un manifeste : il rejoue les manifestes de `su2 boite` et de
`fibre echantillonner`, pas ceux de `su2 gen` : voir « Limites connues ») et **relance manuelle** (la commande `argv` d'un manifeste, relancée dans un dossier neuf).

*In English.* kvu is a Rust command-line tool for lattice gauge theory (mainly 4D SU(2) with the Wilson action): a Monte Carlo
generator with one GPU kernel source for HIP and CUDA plus an FP64 CPU reference, measurement tools (Wilson loops, Polyakov
loops, gradient flow, topology, block-spin RG maps, truncated effective-action couplings, block-face fluxes, 't Hooft twisted
boundary conditions), provenance manifests and verifiable archives. No general bit-reproducibility guarantee is given: the tests of the archive establish, on the CPU (backend `cpu`, FP64, small lattices), that an interrupted and resumed run gives the same files as the uninterrupted run (tests `su2_cpu::reprise_1_standard_nouveau_processus`, `su2_cpu::reprise_1_bord_nouveau_processus`, `su2_torsion::t6_reprise_avec_torsion`, `su2_reprise_mesures::*`), and that on seven CPU runs the compared files of 0.5.1 equal those of the internal 0.5.0; nothing more is established, in particular nothing across backends or machines (replay by `kvu regen` for
`su2 boite` and constrained-fibre runs; for `su2 gen`, rerun the command recorded in the manifest). Documentation is in French.

> **Ce que kvu est, et ce qu'il n'est pas.** kvu est un **instrument**. Il ne contient aucun résultat physique et n'en
> revendique aucun : ni trou de masse, ni limite du continu, ni constante physique. Les modules décrits ci-dessous mesurent
> des quantités définies sur des configurations données. Les tests contrôlent certaines propriétés des mesures sur des cas documentés ; ils ne constituent ni une preuve générale de justesse ni une validation physique.

Version : **0.5.1** (novembre 2026 ; archive identifiée par l'étiquette `v0.5.1` et son SHA-256, donné avec la publication ; les versions 0.2 à 0.5.0 étaient internes et n'ont pas été publiées). Licence : **Apache-2.0** (`LICENSE`, `NOTICE`). Auteur : Kévin Rémondière.
Le code a été écrit par des agents de codage IA sous la direction de l'auteur. Il n'a pas fait l'objet d'une relecture humaine ligne à ligne. (`AI_USE.md`) Taille de l'archive : 1 133 fichiers, tous types confondus (code, tests, documents, scripts).

## Contenu

| sous-commande | rôle |
|---|---|
| `kvu doctor`, `kvu etat` | machine, GPU visibles, backend retenu et pourquoi, politique (verrous, plafonds, garde disque) |
| `kvu su2 gen` | générateur SU(2) Wilson 4D : bain thermique (Creutz / Kennedy–Pendleton) + surrelaxation, damier, Philox4x32-10 à compteur ; modes F64 et F32 (l'option `--options compat-prod` du mode F64 est comparée aux ancres du générateur publié `su2-heatbath-gpu` : voir « État de la validation sur carte (0.5.1) ») ; lots de chaînes, reprise après interruption (identité des fichiers après reprise contrôlée sur processeur par les tests nommés dans l'introduction ; sur carte : voir la même section), mesures en ligne |
| `kvu su2 boite` | boîtes ℓ⁴ à bord fixé (Dirichlet), masque de liens gelés |
| `kvu cond`, `kvu fibre` | bain thermique conditionnel « fin sachant grossier », échantillonnage et mesure de fibres |
| `kvu obs` | boucles de Wilson, corrélateur de Polyakov, flot de gradient (E, t₀, w₀, Q : les conventions complètes (normalisations, unités de réseau) ne sont pas encore rédigées en un seul endroit ; on les lit dans `crates/kvu/GUIDE.md` §« obs » et dans le code de `src/obs/` ; documentation longue reportée), charge topologique |
| `kvu rg` | blocage de Bałaban L_b = 3 et témoin de Swendsen b = 2, domaine, estimateurs de couplage effectif, blocage quasi-local, et les modules M2 et M3 |
| `kvu hess` | hessienne de l'action de Wilson sur un fond donné (H·v, λ_min généralisé par LOBPCG) |
| `kvu groupe`, `kvu gen` | registre de groupes (SU(2), Sp(4)) avec batterie de gates à réponse connue ; générateur générique Sp(4) sur CPU |
| `kvu store`, `kvu fetch`, `kvu regen`, `kvu queue` | store d'ensembles, rapatriement vérifié, régénération depuis un manifeste (`su2 boite`, `fibre echantillonner`, manifestes historiques ; pas `su2 gen` : voir « Limites connues »), file de jobs |
| `kvu su2 thermo` | [EXPÉRIMENTAL] mesures de la méthode intégrale sur boîtes rectangulaires (carte graphique AMD ou NVIDIA) : rapports de torsion, séries, couplages séparés (`--beta-s`), estimateur à lien intégré (`--lien-integre`), coutures décalées sur carte |
| `kvu su2 spectro` | [EXPÉRIMENTAL] spectroscopie de torelons et de tubes de flux sur boîte rectangulaire : opérateurs, impulsions (dont hors axe), blocage, algorithme à deux niveaux, profil du tube, secteurs de flux ; les backends dépendent de la sous-commande : `run` et `neutre` sur processeur, `gpu` et `gpu-torelon` sur carte (HIP ou CUDA, selon `--backend` ; combinaisons et restrictions : `docs/ETAT_v0.3.md` et l'aide de chaque sous-commande) ; état de validation : section « État de la validation sur carte (0.5.1) » |
| `kvu su2 tube` ; `kvu su2 gen --cotes` ; `kvu su2 gen --decalage` | [EXPÉRIMENTAL] `su2 tube` : tube tordu, CPU F64 ; `su2 gen --cotes` : côtés quelconques, CPU F64, et carte (HIP, CUDA ; F64, moteur `une_boite`) selon `docs/ETAT_v0.3.md` ; `su2 gen --decalage` : tores obliques (couture décalée), processeur seulement ; la couture décalée sur carte est une option de `su2 thermo mesure --decalage` (ligne `su2 thermo`). Aucun de ces backends n'est déduit validé : section « État de la validation sur carte (0.5.1) » |
| `kvu intrication` | [EXPÉRIMENTAL] entropie de Rényi-2 par répliques, SU(2) Wilson 4D, CPU F64 |
| `kvu experience` | [EXPÉRIMENTAL] carnet d'expérience : préenregistrements hachés, runs, portes, verdicts, fiche |
| `kvu obs champs` | [EXPÉRIMENTAL] champs par site d'une configuration (pour la visualisation) |
| `kvu valide`, `kvu bench` | suite de validation (gates), bancs de débit |

Quatre modules, déjà listés pour la 0.1.0 (CHANGELOG), sont des **instruments** :

| module | commande | ce qu'il fait | limites |
|---|---|---|---|
| **M1** archive vérifiable | `su2 gen --zstd --archive`, `su2 archive verifier\|indexer\|extraire\|lister` | index `kvu-archive/1` : chaque configuration reliée à son indice, à ses balayages, aux SHA-256 du fichier compressé et du contenu, et à la provenance du run ; le lecteur vérifie avant de rendre un champ | exige le binaire `zstd` du système |
| **M2** action effective tronquée | `rg action`, `rg action-gates` | couplages des boucles plaquette, rectangle, chaise et torsade par pseudo-vraisemblance et par équations de Schwinger–Dyson, erreurs par jackknife ; échantillonneur de Gibbs de contrôle | CPU F64 ; action tronquée à quatre types de boucles, avec ou sans blocage selon `--pas K` : `--pas K` = nombre de blocages de facteur 3 (K = 0 : sans blocage, comme dans l'exemple ci-dessous ; avec blocage, L/3^K ≥ 4) ; lit une chaîne unique de `su2 gen` ou un manifeste de production (une seule chaîne : `--jackknife blocs:B`) ; une action tronquée ne dit rien des termes omis ; dimensions, géométrie, multiplicité des boucles, signe et normalisation des couplages : `docs/ACTION_EFFECTIVE.md` |
| **M3** flux de bloc | `rg flux`, `rg flux-gates` | fraction des faces de bloc n×n dont l'angle de rotation Θ ∈ [0, 2π] de l'holonomie dépasse des seuils prédéfinis, et un indicateur de cohérence des flux de plaquette transportés d'une face, C = Φ_coh/S (Φ_coh : norme de la somme des flux de plaquette transportés au coin, S : somme de leurs angles ; le cas S = 0 et la convention reliant angle et représentant SU(2) : `docs/FLUX_BLOC.md`) | CPU ; L multiple de 3 pour les pavages de blocs |
| **M4** torsion de 't Hooft | `su2 gen --torsion 01,23` | action à plaquettes tordues par le centre ℤ₂ dans les plans choisis (`01,23` : plans (0,1) et (2,3) ; le support exact des facteurs de centre est décrit dans `docs/TORSION.md`), agrafe tordue dans le bain thermique et la surrelaxation | F64 ; `--backend` explicite : `cpu`, `hip` ou `cuda` (« auto » est refusé) ; le contrôle à L = 4 sur CPU (ce qui est comparé, à quelle référence, critère d'acceptation) est décrit dans `docs/TORSION.md` ; les facteurs du centre modifient les plaquettes du support défini dans ce document, pas toutes les plaquettes des orientations citées ; refusée avec `--bord` et `--mesures` |

**Ajouts de la ligne 0.5** (depuis la 0.1.1, détail dans `CHANGELOG.md` et `docs/ETAT_v0.5.md`) : régime de chaque ajout.
Sous-commandes à invoquer explicitement, marquées [EXPÉRIMENTAL] ci-dessus (`su2 thermo`, `su2 spectro`, `su2 tube`, `intrication`, `experience`, `obs champs`, `rg correlateur`, `rg apparier`) : elles ne s'exécutent que sur invocation ; leur niveau de validation sur carte est celui de la section « État de la validation sur carte (0.5.1) » (aucune déduction d'un backend annoncé à un backend testé).
Options **éteintes par défaut** (`su2 gen --cotes`, `--decalage`, `--torsion`, `--beta-s`, `--lien-integre`, `rg action --erreurs bootstrap`, mesure en ligne `corr`) : sans elles, sur sept essais processeur, les fichiers comparés de la 0.5.1 sont identiques au bit à ceux de la 0.5.0 interne (`docs/PUBLICATION.md` §5 : fichiers comparés et exclus).
Outils Python de lecture, lancés à la main (`scripts/thermo/`, `scripts/spectro/`, `scripts/tube/`, `scripts/lecture_graine/`). Les tests contrôlent certaines propriétés sur des cas documentés, pas la signification des quantités.
Les méthodes citent leurs sources dans les notes de `docs/v0.2/notes/` et dans le code ; l'estimateur à lien intégré est un estimateur multi-hit par intégration analytique de la moyenne conditionnelle d'un lien, dans les conditions indiquées (G. Parisi, R. Petronzio, F. Rapuano, *Phys. Lett. B* 128 (1983) 418), sans promesse d'exactitude du résultat Monte-Carlo.

Documentation : `crates/kvu/GUIDE.md` (mode d'emploi complet ; ses commandes sont vérifiées par un test), `docs/` (un fichier par
module : `ARCHIVE.md`, `ACTION_EFFECTIVE.md`, `FLUX_BLOC.md`, `TORSION.md`, `BOITES.md`, `MACHINES.md`, `CONVENTIONS.md`…),
`crates/kvu/ARCHITECTURE_UNIFIEE.md` (conception), `CHANGELOG.md`, `docs/PUBLICATION.md` (ce qui a été retiré ou neutralisé
pour la publication).

## Installation

Prérequis :

- Linux x86_64 ; **Rust 1.97.0** (épinglé par `rust-toolchain.toml` : `rustup` l'installe tout seul) ;
- pour la suite de tests complète : **Python 3** avec **numpy** et **scipy** (des tests appellent des scripts de référence
  indépendants écrits en Python), et `bash`, `git`, `zstd` (≥ 1.3.4 ; facultatif pour un run ordinaire, requis par `--zstd` et par l'exemple M1 ci-dessous) ;
- pour le GPU, à l'exécution seulement : ROCm (`libamdhip64`, `libhiprtc`) pour HIP, ou le pilote NVIDIA et NVRTC (CUDA ≥ 12.8)
  pour CUDA. Aucune bibliothèque GPU n'est nécessaire à la compilation.

Parcours CPU pas à pas, depuis l'archive de l'étiquette `v0.5.1` (`kvu-0.5.1.zip`, dont le SHA-256 est donné avec la publication) :

```bash
unzip kvu-0.5.1.zip && cd kvu-0.5.1                      # ou : git clone, puis git checkout v0.5.1
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # seulement si rustup est absent ; rust-toolchain.toml épingle Rust 1.97.0
cargo build --release --locked -p kvu                    # target/release/kvu ; backend HIP toujours compilé (dlopen)
target/release/kvu --version                             # attendu : « kvu 0.5.1 » suivi de l'empreinte des sources
```

Option CUDA (dépendance GPU, non nécessaire au parcours CPU) :

```bash
cargo build --release --locked -p kvu --features cuda    # ajoute le backend CUDA (cudarc, chargement dynamique)
```

Pour taper `kvu` au lieu de `target/release/kvu`, depuis la racine du dépôt : `export PATH="$PWD/target/release:$PATH"` (les exemples
ci-dessous écrivent `kvu`). Premier contrôle : `kvu --version`, puis `kvu doctor` (sans fichier de configuration, `Machine : inconnue`
est normal ; `GPU : aucun` signifie calcul sur le CPU). `kvu doctor --test` essaie un noyau sur le GPU.

Configuration des machines (verrous GPU, plafonds mémoire, garde disque, store) : copier le **modèle** `kvu.toml` dans
`~/.config/kvu/kvu.toml` et l'adapter (`docs/MACHINES.md`). Sans fichier, toute machine prend le profil prudent `inconnue`.

## Exemples minimaux (CPU)

Exemples CPU pour l'environnement décrit ci-dessus, à lancer dans un répertoire de travail neuf. M1 exige `zstd`. `scripts/exemples_readme.sh` permet de les lancer telles quelles, dans
un dossier jetable (c'est une étape de `scripts/recette_locale.sh`).

<!-- exemples:debut -->
```bash
kvu --version
kvu su2 selftest                                   # vecteurs de référence de Philox4x32-10
kvu doctor                                         # machine, GPU visibles, backend retenu, politique
# un petit ensemble SU(2) (L = 6, β = 2,3) sur le CPU en F64, avec un dump tous les 10 indices de mesure
kvu su2 gen --backend cpu --L 6 --beta 2.3 --graine 1 --start cold --therm 20 --nmeas 20 --nor 1 --dump-every 10 --format q64 --outdir ens_L6
kvu obs wilson --backend cpu --cfg ens_L6/cfg_000020.npy --out wilson_L6 --rmax 2 --tmax 2
# M1 : dumps compressés et archive vérifiable
kvu su2 gen --backend cpu --L 4 --beta 2.3 --graine 2 --start cold --therm 10 --nmeas 10 --nor 1 --dump-every 5 --format q64 --zstd --archive --outdir arch_L4
kvu su2 archive verifier arch_L4
kvu su2 archive extraire arch_L4 --indice 10 --sortie arch_L4_cfg10.npy
# M2 : couplages de l'action effective tronquée (sans blocage) sur l'ensemble L = 6
kvu rg action --ensemble ens_L6 --jackknife blocs:2 --sortie action_L6
# M3 : faces de bloc 3×3 et flux cohérent
kvu rg flux --entree ens_L6/cfg_000020.npy --n 1,2,3 --coherent 3 --sortie flux_L6.json
# M4 : torsion de 't Hooft dans les plans (0,1) et (2,3)
kvu su2 gen --backend cpu --L 4 --beta 2.3 --graine 3 --start cold --therm 10 --nmeas 10 --nor 1 --torsion 01,23 --outdir tordu_L4
```
<!-- exemples:fin -->

**Un calcul complet** (mesure en ligne, interruption propre, reprise, puis lecture des résultats) : exemple
`EXEMPLE_MODE=petit EXEMPLE_SORTIE=$PWD/essai KVU_BIN=$PWD/target/release/kvu crates/kvu/exemples/05_mesure_enligne_reprise.sh`
(sans `EXEMPLE_SORTIE`, les sorties sont effacées à la fin du script ; le dossier `essai` doit être neuf ; l'exemple interrompt proprement le run puis le reprend, les arrêts brutaux étant éprouvés par les tests du dépôt), puis
`crates/kvu/GUIDE.md` §3 « su2 gen » (options, `--mesures`, fichiers produits, calcul d'une moyenne avec son erreur). Si une commande
répond `garde SSD … REFUS`, la place libre moins la taille de l'écriture est inférieure au seuil de la machine (20 Go par défaut) :
choisir d'abord un volume qui a la place ; `KEVINOTRON_SSD_MIN_GO` abaisse le seuil (variable héritée, voir « Noms hérités ») et n'est à employer que pour une recette contrôlée. `kvu experience` consigne un calcul dans un carnet vérifiable (GUIDE §13).

Dans l'exemple périodique ci-dessus (`ens_L6`), lancé dans un dossier neuf et mené à terme, `su2 gen` écrit `measurements.csv`, `therm.csv`, `manifest.json` (provenance complète), les dumps
`cfg_NNNNNN.npy[.zst]` et, en dernier, `COMPLETE` ; d'autres sorties dépendent des options (`--bord`, `--mesures`, `--archive`), et `COMPLETE` n'existe ni après un échec ni après une interruption. Les sorties de `rg action` et `rg flux` sont des JSON (`action.json`,
`flux_L6.json`). Colonnes de `measurements.csv` : `measurement_index` (indice de la mesure ; 0 = configuration initiale), `total_steps` (nombre total de balayages effectués), `phase` (`initial` ou `sample`), `plaquette` (moyenne de ½ Re Tr U_p sur les six plans et tous les sites, sans unité ; 1 pour la configuration froide initiale), `P0` à `P3` (boucle de Polyakov moyenne, ½ Re Tr, le long des directions 0 à 3 ; PAS des plaquettes par plan) ; `therm.csv` : `step,plaquette,P0,P1,P2,P3`. Contrôle numérique public : à l'indice 0 de `ens_L6/measurements.csv` (départ froid), la plaquette et les quatre Polyakov valent 1. Chaque commande ci-dessus rend le code 0 sur la machine de la recette (journal `preuves/exemples_051c.out`, ci-après) ; le code 0 de `kvu su2 archive verifier arch_L4` signifie qu'aucun défaut de cohérence de l'archive n'est trouvé, pas que les mesures sont justes. Ces petits ensembles ne sont ni thermalisés ni décorrélés : ils servent à contrôler l'installation, pas à mesurer ; thermalisation, autocorrélation et erreur statistique d'un vrai calcul : `crates/kvu/GUIDE.md` §3. Le détail des options et des formats est dans `crates/kvu/GUIDE.md`.

## Tests et recette

```bash
cargo test --release --locked -p kvu                # suite par défaut, hors tests ignorés (les tests de carte sont #[ignore])
scripts/recette_locale.sh                           # build, suite par défaut, contrôle des noms de tests, clippy, build CUDA --no-run
```

Mesure de référence pour cette version (`git archive` de la version, dépendances téléchargées depuis crates.io avec `--locked`, aucun
accès aux cartes graphiques, `HOME` vide, `KEVINOTRON_SSD_MIN_GO=0`, 4 tâches de compilation et 4 fils ; PC AMD Ryzen AI Max+ 395) :
**2073 tests listés dans 116 exécutables : 1861 actifs, tous passés, 0 échec, 212 ignorés** (ventilation des 212 : section « État de la validation sur carte (0.5.1) ») ; les résumés de cargo additionnent 1 866 « passed », soit 5 de plus que les 1 861 tests actifs listés (l'origine de l'écart n'est pas établie par un journal ; le décompte des tests est celui de `cargo test --release --locked -p kvu -- --list`). Détail dans `CHANGELOG.md` et `docs/PUBLICATION.md`. Clippy : `cargo clippy --locked -p kvu --all-targets -- -D warnings` est **vert**, avec et sans `--features cuda` ; la compilation avec `--features cuda` (`cargo test --release --locked -p kvu --features cuda --no-run`) réussit : c'est une compilation, pas une exécution sur carte. La 0.1.0 comptait 1463 tests actifs et 115 ignorés (liste `v0.1.0.cpu.txt`).

`scripts/recette/noms_tests/v0.5.1.cpu.txt` est la liste des tests telle que cargo la compte pour cette version (`v0.1.0.cpu.txt`, celle
de la 0.1.0, est conservée : aucun de ses tests n'a été retiré) : la recette refuse un candidat qui en retire ou en désactive un. Les **tests de carte** (en majorité marqués `#[ignore = "GPU"]`) exigent une carte, le verrou du
profil et, pour certains, des ensembles de référence absents de l'archive publique (voir « État de la validation sur carte (0.5.1) ») . Ils ne forment pas une recette publique : `cargo test --release --locked -p kvu -- --ignored` sélectionne aussi les ignorés non GPU (lourds, SSH, passerelles), et `--features cuda` compile le backend CUDA sans sélectionner à lui seul les essais NVIDIA. La méthode utilisée pour la recette des cartes (liste exacte des 181 tests de carte, un test par appel, verrou de carte) est décrite dans la section « État de la validation sur carte (0.5.1) » ; les tests qui exigent des références absentes de l'archive y sont comptés à part. Les gates de `crates/kvu/validation/` documentent la validation faite sur les deux machines de l'auteur et
supposent ses données : elles ne tournent pas telles quelles ailleurs.

## État de la validation sur carte (0.5.1)

Bilan détaillé de référence ; les autres textes y renvoient. **Mesuré le 24 novembre 2026** sur les deux machines de l'auteur, à partir de l'archive (binaires de test construits par `cargo test --release --locked -p kvu --no-run`, avec `--features cuda` sur NVIDIA ; empreinte de source `kvu-source/1` du binaire essayé : `e50cc304…` ; l'archive finale ne diffère de celle des essais que par des documents (`README.md`, `CHANGELOG.md`, `AI_USE.md`, `CITATION.cff`, `.zenodo.json`, `docs/`), par la liste de référence des noms de tests, par une ligne de commentaire de documentation dans `reprise.rs`, par le champ `authors` de `crates/kvu/Cargo.toml` (l'adresse de messagerie retirée) et par les motifs d'ignorance écrits de quatre tests `g_bit_*` : hors cette ligne de commentaire, aucune ligne de `crates/kvu/src` ni des noyaux n'a changé depuis les essais sur carte, qui n'ont pas été rejoués sur carte après ces changements de métadonnées et de texte).
Chaque test de carte a été lancé SEUL (`<binaire de test> --ignored --exact <nom>`, délai de 110 s par test, 280 s pour les tests lents relancés à part), sous le verrou de carte de la machine, pendant que des productions de mesure tournaient sur les mêmes cartes (ce qui explique des délais d'attente, et la mémoire occupée sur NVIDIA). Les outils qui ont servi (`liste_tests_carte.py`, `recette_cartes_lot.sh`, `bilan_cartes.py`) sont ceux du paquet de préparation, non publiés ; la liste de 181 tests est celle de `scripts/ventilation_ignores.py` (catégories « carte »).

- **AMD : Radeon 8060S (gfx1151), HIP**, bibliothèques HIP et HIPRTC système (`libhiprtc.so.7`), noyau Linux 7.0.0-38. **NVIDIA : GeForce RTX 5060 Ti (sm_120), CUDA**, pilote 595.99.02 (CUDA 13.2), NVRTC 13, noyau Linux 7.0.0-43.
- **Temps de carte** consommé par les essais (hors attente du verrou) : AMD 33 min 51 s (somme des durées des tests, démarrages compris, hors attente du verrou ; budget : 60 min) ; NVIDIA 30 min 2 s (somme des durées des tests, démarrages compris, hors attente du verrou ; budget : 60 min).

| résultat de chaque test | AMD (HIP) | NVIDIA (CUDA) |
|---|---|---|
| tests de carte listés et lancés | 181 | 181 |
| réussi | 149 | 155 |
| non conclu : délai dépassé | 3 | 1 |
| non concluant : mémoire de la carte occupée par une production (out of memory) | 0 | 1 |
| rendu « ok » en s'écartant lui-même, en tout ou en partie (ligne SKIP) | 12 | 11 |
| réussi, durée < 0,05 s : calcul non établi | 7 | 4 |
| sans objet sur cette carte (test cuda) | 7 | 0 |
| sans objet sur cette carte (test hip) | 0 | 6 |
| échoué : exige des références non publiées | 1 | 1 |
| échoué : gate historique déclassée (PUB, OBS-4-L16) | 1 | 1 |
| échoué : non concluant aux paramètres par défaut (campagne F3 de production) | 1 | 1 |


- **Lecture des lignes** : « réussi » = le test a rendu 0 et affiche « 1 passed », sans ligne `SKIP` ; « rendu ok en s'écartant lui-même » = le test contient une ligne `SKIP` (références, variable d'environnement ou condition absentes) : il ne vaut PAS preuve, en tout ou en partie ; « réussi, durée < 0,05 s » = rendu ok presque instantanément, calcul non établi ; « sans objet sur cette carte » = test propre à l'autre backend (HIP sur NVIDIA, CUDA sur AMD) : ce n'est pas un échec du code ; « non conclu » = délai dépassé.
- **Échecs** : `su2_gpu::g_bit_hip_prod` (AMD) et `su2_gpu::g_bit_cuda` (NVIDIA) exigent des références de l'auteur absentes de l'archive (« référence absente ») : ils échoueront chez tout lecteur ; ils sont comptés à part et le texte ne les compte pas comme preuve. `obs_gpu::obs_4_l16_quantification` échoue sur les deux cartes (5 configurations sur 20 à moins de 0,1 d'un entier, contre 18 requises) : c'est la gate historique déclassée (`docs/DEFAUTS_CONNUS.md`). `une_boite::f3_verdict` est non concluant aux paramètres par défaut (σ de l'ordre de 4·10⁻⁵ contre 3·10⁻⁷ exigés : la porte F3 demande une campagne de production, variables `KVU_F3_*`). `fibre_gpu::banc_fibre` (NVIDIA) échoue par « CUDA_ERROR_OUT_OF_MEMORY » : la mémoire de la carte (16 Go) est en grande partie occupée par une production de mesure (environ 13 Go) ; non concluant, à rejouer sur une carte libre.
- **Délai dépassé (non conclu, 280 s)** : `impairs_1120::g_plaquette_impairs_contre_pairs` sur les deux cartes, `une_boite::g5_methode_directe_l2` et `une_boite::g8_mutants` sur AMD (les mêmes réussissent sur NVIDIA ou sont plus longs que le créneau) ; un délai plus long n'a pas été essayé (appels de 300 s au plus par consigne de partage des cartes).
- **Seconde passe** : les tests qui lisent le backend ou la machine dans l'environnement (`KVU_TEST_BACKEND`, `KEVINOTRON_BACKEND`, `KVU_F32_SOUS_VERROU`, `KVU_TEST_MACHINE_ATTENDUE`) ont été relancés avec le backend et la machine de la carte essayée ; le résultat de la seconde passe remplace celui de la première.
- **Tests de reprise sur carte** (réussis sur les deux cartes) : `su2_gpu::{reprise_1_f64, reprise_1_bord, reprise_1_f32, f32_reprise_identite}`, `su2_corr_enligne::{e5_gpu_reprise, e5_gpu_reprise_sans_mesures_recharge, v3_gpu_provenance_apres_deux_reprises}`, `cond_gpu::gate_eqc4_reprise_de_chaine_a_l_octet_gpu`, `geo_gpu::o1_reprise_gpu_et_invariants`.
- **Ancre du générateur publié, sur AMD seulement** : `kvu su2 gen --backend hip --precision f64 --options compat-prod --L 12 --beta 2.5 --graine 26095612011 --therm 5000 --nmeas 2000 --dump-every 100 --start hot --skip 1 --nor 4 --outdir SORTIE`, binaire de cette version (`kvu 0.5.1`, SHA-256 `c2b1633e…`), sur la Radeon 8060S : `cfg_002000.npy`, `measurements.csv` et `therm.csv` ont les trois empreintes SHA-256 des lignes `hip` de `tests/anchors.sha256` de `su2-heatbath-gpu` (DOI 10.5281/zenodo.23644087), recopiées dans `docs/ancres_su2_heatbath_gpu.sha256` ; 3 fichiers sur 3, 10 s ; journal : `preuves/ancre_hip_051c.log` du paquet de préparation. Établi pour ce cas, pour le mode F64 sur HIP ; non établi sur CUDA (les lignes `cuda` de ce fichier n'ont pas été comparées), ni en F32.
- **Les 212 tests ignorés** se ventilent ainsi (`#[ignore = "…"]`, recomptés par `python3 scripts/ventilation_ignores.py .`) : 181 tests de carte (167 marqués « GPU », 13 comparaisons carte-processeur sans motif écrit, 1 enfant d'une gate de carte) ; 1 test qui exige l'accès SSH à une seconde machine de l'auteur ; 2 passerelles lancées par des scripts de validation ; 1 test qui exige une copie de données (`KVU_EXP_RETRO_DIR`) ; 27 tests lourds, manuels ou d'exploration sur processeur. Les 31 qui ne sont pas des tests de carte n'ont pas été rejoués.
- **Identité au bit.** Les sorties AMD et NVIDIA ne sont pas identiques au bit. Aucun accord statistique AMD–NVIDIA n'est établi ici pour la 0.5.1.

## Limites connues

- **Plateformes** : processeur Linux x86_64 (Ubuntu 26.04), rejoué depuis l'archive. Backends proposés : HIP (AMD) et CUDA (NVIDIA) ; machines de
  développement : AMD Radeon 8060S (gfx1151, ROCm 7.1) et NVIDIA GeForce RTX 5060 Ti (sm_120, CUDA 13). Leur état de validation pour la 0.5.1 est dans
  [« État de la validation sur carte (0.5.1) »](#état-de-la-validation-sur-carte-051) ; autres cartes : non testées.
- **Tests ignorés** : voir la ventilation des 212 tests ignorés dans la section « État de la validation sur carte (0.5.1) » ; les gates des modules (`rg action-gates`, `rg flux-gates`) ne font pas partie de la suite par défaut.
- **Charge topologique de `obs flot` : expérimental, défaut connu.** Dans le test historique `OBS-4-L16` (gate GPU), à L = 16, β = 2,5 et t/a² = 3, 5 configurations sur 20 satisfaisaient le critère de proximité à 0,1 d'un entier, contre 18 requises ; test de la 0.1.0, non rejoué pour la 0.5.1 ;
  la gate est déclassée, son seuil n'a pas été changé ; aucune conclusion générale à ce β n'en est tirée. `kvu obs topo` utilise une définition améliorée ; l'intégralité n'est pas garantie. Détail : `docs/DEFAUTS_CONNUS.md`.
- **M2** n'estime qu'une action tronquée à quatre boucles et un pas de blocage ; **M4** : étalonnage CPU à L = 4 seulement ;
  **M3** suppose L multiple de 3. Aucun de ces modules ne borne ce qu'il ne mesure pas.
- **Backends par option** : la matrice complète commande × option × backend × précision n'est pas publiée ; les refus d'options incompatibles (code 2) sont décrits dans `crates/kvu/GUIDE.md` et dans l'aide de chaque commande (`--help`). Ne pas déduire d'une option listée plus haut qu'elle est validée sur carte : voir la section « État de la validation sur carte (0.5.1) ».
- **Noms hérités** : les profils de machines s'appellent `acemagic` et `gamer` (les deux machines de référence), et les
  variables d'environnement gardent le préfixe `KEVINOTRON_` du moteur interne dont kvu est issu. Les renommer changerait
  l'interface et le format des manifestes : annoncé pour la 0.2, **non fait dans la 0.5.1** ; reporté à la 0.6, avec des alias pour
  les anciens noms.
- **Le fichier `crates/kvu/kvu.toml` est une fixture de test** qui reconnaît les machines de l'auteur par leur GPU (gfx1151,
  sm_120) : lancer `kvu` depuis `crates/kvu/` appliquerait ces profils. Utiliser le modèle de la racine.
- **Compression** : sous un plafond de tâches (`TasksMax`), zstd n'est lancé que sans fil auxiliaire (`--single-thread`, et
  `--no-asyncio` si le zstd installé le connaît, ≥ 1.5.4).
- **Régénération** : `kvu regen` prend un manifeste `kvu-regen/1` (produit par `su2 boite` et `fibre echantillonner` sous le nom `DIR/manifest_regen.json`) et rejoue la commande qu'il enregistre. Le seul test qui établit un rejeu identique dans l'archive est `regen_cpu::regen_cpu_rejeu_identique_et_copie_du_manifeste` : une chaîne `su2 gen` jouet (CPU, F64, `ref-k1`, L = 4, 7 fichiers) décrite par un manifeste `kvu-regen/1` construit par le test, rejouée par le même binaire. Critère d'admission : le SHA-256 du binaire exécutant égale celui du manifeste, ou une preuve G-BIT (`--preuve-gbit`) existe pour le couple (backend, options) ; sinon « binaire non admis » (code 2, test `regen_cpu_binaire_non_admis_est_refuse_sans_rien_ecrire`). Les manifestes historiques `su2_hip/1` et `su2_gen/1` sont lus par le classement de `kvu regen` ; leur rejeu sur carte n'est pas testé dans l'archive. Distinction : une lecture réussie d'un manifeste n'est pas un rejeu réussi. `kvu regen` ne rejoue PAS un run `su2 gen` ordinaire : son `manifest.json` n'est pas reconnu (code 4) et le `regen_bloc3.json` de `--mesures bloc:3`
  échoue au rejeu (code 8) ; défaut PUB-15 de `docs/DEFAUTS_CONNUS.md`. Pour un run `su2 gen`, la relance manuelle consiste à relancer la commande enregistrée dans
  `manifest.json` (clé `argv`), avec le même binaire et le même backend sur la même machine, dans un dossier `--outdir` neuf et
  sans `--arret-apres` ni `--fenetre-s`, puis à comparer les sha256 des fichiers à ceux du manifeste (commandes :
  `crates/kvu/GUIDE.md`, « Régénération des ensembles ») ; aucun test de l'archive ne l'établit en général, seuls les cas ci-dessus et les essais CPU de l'introduction sont couverts.
- **Modules [EXPÉRIMENTAL] de la ligne 0.5** : chacun a ses limites dans `docs/ETAT_v0.5.md` (« Mauvaises nouvelles ») et
  `docs/ETAT_v0.3.md`, `docs/ETAT_v0.4.md` ; en particulier, les sorties AMD et NVIDIA ne sont pas identiques au bit ; aucun accord statistique AMD–NVIDIA n'est établi ici pour la 0.5.1 ; les tests de carte ne sont pas rejoués par la recette locale.
- Documentation et messages en français.

## Historique et nom (History and naming)

- kvu est le successeur nettoyé d'un outillage interne de l'auteur dont le nom de travail était « kevinotron » ; d'où les
  noms `KEVINOTRON_*` (et quelques identifiants internes), encore présents dans la 0.5.1 (aucun changement de code à ce sujet dans cette version ; renommage avec alias des anciens noms prévu pour la 0.6, voir `CHANGELOG.md`).
- Deux dépôts Zenodo exploratoires de mai 2026 portent ce nom : « Kevinotron v9.0.0 »
  ([10.5281/zenodo.20412608](https://doi.org/10.5281/zenodo.20412608)) et « Crossed Cosmos / Kevinotron v10.0.0 »
  ([10.5281/zenodo.20415354](https://doi.org/10.5281/zenodo.20415354)), versions du même concept 10.5281/zenodo.20379360.
  Ils précèdent kvu ; la version v10.0.0 porte un avis de correction daté du 12 juillet 2026. Leur contenu n'est pas un
  résultat de kvu, et kvu ne le reprend pas.
- kvu ne contient aucune affirmation de physique : ce sont des instruments.

*In English.* kvu is the cleaned-up successor of the author's internal tooling, whose working name was "kevinotron" (hence
the `KEVINOTRON_*` names, still present in 0.5.1). The exploratory Zenodo records of May 2026 bearing that name (10.5281/zenodo.20412608,
10.5281/zenodo.20415354; the latter carries a correction notice dated 12 July 2026) predate kvu and are not results of kvu.
kvu makes no physics claim: it provides instruments.

## Logiciel lié, citation, licence

- Générateur SU(2) d'origine (le mode F64 `compat-prod` de kvu est comparé à ses ancres : voir « État de la validation sur carte (0.5.1) ») : **su2-heatbath-gpu**, DOI
  [10.5281/zenodo.23644087](https://doi.org/10.5281/zenodo.23644087).
- Pour citer kvu : `CITATION.cff`. Citation : Kévin Rémondière, *kvu* 0.5.1, logiciel, novembre 2026, Apache-2.0, https://github.com/AIdevsmartdata/kvu (tag `v0.5.1`, archive `kvu-0.5.1.zip` de cette étiquette ; la date de recette n'est pas la date de dépôt), DOI de version [10.5281/zenodo.24544352](https://doi.org/10.5281/zenodo.24544352), distinct du DOI de toutes versions 10.5281/zenodo.24062151 (l'archive déposée ne contient pas son propre DOI, ajouté ensuite dans ce README et dans `CITATION.cff`). Zenodo : version 0.5.1 [10.5281/zenodo.24544352](https://doi.org/10.5281/zenodo.24544352), version 0.1.1 [10.5281/zenodo.24151227](https://doi.org/10.5281/zenodo.24151227), version 0.1.0 [10.5281/zenodo.24062152](https://doi.org/10.5281/zenodo.24062152), toutes versions [10.5281/zenodo.24062151](https://doi.org/10.5281/zenodo.24062151).
- Logiciels voisins : la théorie de jauge sur réseau dispose de codes établis et bien plus généraux, par exemple QUDA (cartes NVIDIA et AMD), Grid, SIMULATeQCD (plusieurs cartes), MILC, openQCD et HiRep (théories SU(N)) ; en Rust, lattice-qcd-rs (jauge pure) ; kvu n'en reprend pas le code et sert un usage plus restreint (SU(2) avec l'action de Wilson, et les outils de mesure de ce projet). *Related software: established and far more general lattice gauge theory codes exist, for example QUDA (NVIDIA and AMD GPUs), Grid, SIMULATeQCD (multi-GPU), MILC, openQCD and HiRep (SU(N) theories); in Rust, lattice-qcd-rs (pure gauge); kvu does not reuse their code and serves a narrower purpose (SU(2) with the Wilson action, and this project's measurement tools).*
- Licence Apache-2.0 ; dépendances et leurs licences : `LICENCES_TIERCES.md`.
