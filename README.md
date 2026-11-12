# kvu

**kvu** est un générateur et un banc de mesure pour les théories de jauge sur réseau, centré sur la théorie de Yang–Mills SU(2)
à action de Wilson en quatre dimensions. Un seul code de noyaux GPU sert deux backends, HIP (AMD) et CUDA (NVIDIA), compilés à
l'exécution, à côté d'une référence CPU en double précision. Chaque exécution écrit un manifeste de provenance (binaire, commit,
empreinte des sources, graines, options), et les configurations peuvent être archivées et vérifiées ; une génération est
déterministe (même binaire, même backend, même machine : mêmes octets), et `kvu regen` rejoue au bit près les manifestes de `su2 boite` et de
`fibre echantillonner` (pas encore ceux de `su2 gen` : voir « Limites connues »).

*In English.* kvu is a Rust command-line tool for lattice gauge theory (mainly 4D SU(2) with the Wilson action): a Monte Carlo
generator with one GPU kernel source for HIP and CUDA plus an FP64 CPU reference, measurement tools (Wilson loops, Polyakov
loops, gradient flow, topology, block-spin RG maps, truncated effective-action couplings, block-face fluxes, 't Hooft twisted
boundary conditions), provenance manifests, verifiable archives and deterministic runs (bit-exact replay by `kvu regen` for
`su2 boite` and constrained-fibre runs; for `su2 gen`, rerun the command recorded in the manifest). Documentation is in French.

> **Ce que kvu est, et ce qu'il n'est pas.** kvu est un **instrument**. Il ne contient aucun résultat physique et n'en
> revendique aucun : ni trou de masse, ni limite du continu, ni constante nouvelle. Les modules décrits ci-dessous mesurent
> des quantités définies précisément sur des configurations données ; leurs tests vérifient qu'ils les mesurent bien
> (réponses connues, références indépendantes, mutants), pas ce que ces quantités signifient.

Version : **0.1.1** (novembre 2026). Licence : **Apache-2.0** (`LICENSE`, `NOTICE`). Auteur : Kévin Rémondière.
Le code et la documentation ont été écrits par des agents de codage IA sous la direction de l'auteur (`AI_USE.md`).

## Contenu

| sous-commande | rôle |
|---|---|
| `kvu doctor`, `kvu etat` | machine, GPU visibles, backend retenu et pourquoi, politique (verrous, plafonds, garde disque) |
| `kvu su2 gen` | générateur SU(2) Wilson 4D : bain thermique (Creutz / Kennedy–Pendleton) + surrelaxation, damier, Philox4x32-10 à compteur ; F64 de référence ou F32 rapide (en F64 avec `--options compat-prod`, sorties GPU bit-identiques à celles du générateur publié `su2-heatbath-gpu`) ; lots de chaînes, reprise au bit, mesures en ligne |
| `kvu su2 boite` | boîtes ℓ⁴ à bord fixé (Dirichlet), masque de liens gelés |
| `kvu cond`, `kvu fibre` | bain thermique conditionnel « fin sachant grossier », échantillonnage et mesure de fibres |
| `kvu obs` | boucles de Wilson, corrélateur de Polyakov, flot de gradient (E, t₀, w₀, Q), charge topologique |
| `kvu rg` | blocage de Bałaban L_b = 3 et témoin de Swendsen b = 2, domaine, estimateurs de couplage effectif, blocage quasi-local, et les modules M2 et M3 |
| `kvu hess` | hessienne de l'action de Wilson sur un fond donné (H·v, λ_min généralisé par LOBPCG) |
| `kvu groupe`, `kvu gen` | registre de groupes (SU(2), Sp(4)) avec batterie de gates à réponse connue ; générateur générique Sp(4) sur CPU |
| `kvu store`, `kvu fetch`, `kvu regen`, `kvu queue` | store d'ensembles, rapatriement vérifié, régénération depuis un manifeste (`su2 boite`, `fibre echantillonner`, manifestes historiques ; pas `su2 gen` : voir « Limites connues »), file de jobs |
| `kvu valide`, `kvu bench` | suite de validation (gates), bancs de débit |

Les quatre modules ajoutés en dernier (novembre 2026) sont des **instruments** :

| module | commande | ce qu'il fait | limites |
|---|---|---|---|
| **M1** archive vérifiable | `su2 gen --zstd --archive`, `su2 archive verifier\|indexer\|extraire\|lister` | index `kvu-archive/1` : chaque configuration reliée à son indice, à ses balayages, aux SHA-256 du fichier compressé et du contenu, et à la provenance du run ; le lecteur vérifie avant de rendre un champ | exige le binaire `zstd` du système |
| **M2** action effective tronquée | `rg action`, `rg action-gates` | couplages des boucles plaquette, rectangle, chaise et torsade par pseudo-vraisemblance et par équations de Schwinger–Dyson, erreurs par jackknife ; échantillonneur de Gibbs de contrôle | CPU F64 ; un pas de blocage (L/3^K ≥ 4) ; lit une chaîne unique de `su2 gen` ou un manifeste de production (une seule chaîne : `--jackknife blocs:B`) ; une action tronquée à quatre boucles ne dit rien des termes omis |
| **M3** flux de bloc | `rg flux`, `rg flux-gates` | fraction des faces de bloc n×n dont l'holonomie dépasse des seuils, et flux cohérent d'une face (aligné ou accidentel) | CPU ; L multiple de 3 pour les pavages de blocs |
| **M4** torsion de 't Hooft | `su2 gen --torsion 01,23` | action à plaquettes tordues par le centre ℤ₂ dans les plans choisis, agrafe tordue dans le bain thermique et la surrelaxation | CPU F64 seulement ; étalonné à L = 4 |

Documentation : `crates/kvu/GUIDE.md` (mode d'emploi complet ; ses commandes sont vérifiées par un test), `docs/` (un fichier par
module : `ARCHIVE.md`, `ACTION_EFFECTIVE.md`, `FLUX_BLOC.md`, `TORSION.md`, `BOITES.md`, `MACHINES.md`, `CONVENTIONS.md`…),
`crates/kvu/ARCHITECTURE_UNIFIEE.md` (conception), `CHANGELOG.md`, `docs/PUBLICATION.md` (ce qui a été retiré ou neutralisé
pour la publication).

## Installation

Prérequis :

- Linux x86_64 ; **Rust 1.97.0** (épinglé par `rust-toolchain.toml` : `rustup` l'installe tout seul) ;
- pour la suite de tests complète : **Python 3** avec **numpy** et **scipy** (des tests appellent des scripts de référence
  indépendants écrits en Python), et `bash`, `git`, `zstd` (≥ 1.3.4 ; la compression des dumps est facultative) ;
- pour le GPU, à l'exécution seulement : ROCm (`libamdhip64`, `libhiprtc`) pour HIP, ou le pilote NVIDIA et NVRTC (CUDA ≥ 12.8)
  pour CUDA. Aucune bibliothèque GPU n'est nécessaire à la compilation.

```bash
cargo build --release --locked -p kvu                    # target/release/kvu ; backend HIP toujours compilé (dlopen)
cargo build --release --locked -p kvu --features cuda    # ajoute le backend CUDA (cudarc, chargement dynamique)
```

Configuration des machines (verrous GPU, plafonds mémoire, garde disque, store) : copier le **modèle** `kvu.toml` dans
`~/.config/kvu/kvu.toml` et l'adapter (`docs/MACHINES.md`). Sans fichier, toute machine prend le profil prudent `inconnue`.

## Exemples minimaux (CPU, quelques secondes)

Ces commandes tournent sur n'importe quelle machine, sans GPU ; `scripts/exemples_readme.sh` les exécute telles quelles, dans
un dossier jetable (c'est une étape de la recette).

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

Chaque `su2 gen` écrit dans un dossier **neuf** `measurements.csv`, `manifest.json` (provenance complète), les dumps
`cfg_NNNNNN.npy[.zst]` et, en dernier, `COMPLETE`. Les sorties de `rg action` et `rg flux` sont des JSON (`action.json`,
`flux_L6.json`). Le détail des options et des formats est dans `crates/kvu/GUIDE.md`.

## Tests et recette

```bash
cargo test --release --locked -p kvu                # suite complète ; les tests GPU sont #[ignore] et ne tournent pas
scripts/recette_locale.sh                           # build, suite complète, contrôle des noms de tests, clippy, build CUDA --no-run
```

Mesure de référence pour cette version (clone propre, dépendances téléchargées depuis crates.io avec `--locked`, aucun accès
au GPU, nom d'hôte neutre, `HOME` vide ; PC Intel Core i5-14600KF, 8 fils) : **1578 tests compilés, 1463 actifs, tous passés,
115 ignorés**, en environ 15 minutes compilation comprise ; détail dans `CHANGELOG.md`, section 0.1.0.

`scripts/recette/noms_tests/v0.1.0.cpu.txt` est la liste des tests telle que cargo la compte pour cette version : la recette
refuse un candidat qui en retire ou en désactive un. Les **tests GPU** (marqués `#[ignore = "GPU"]`) exigent un GPU, le verrou du
profil et souvent des ensembles de référence ; ils se lancent explicitement (`cargo test --release -p kvu -- --ignored`, sur une
machine dédiée). Les gates de `crates/kvu/validation/` documentent la validation faite sur les deux machines de l'auteur et
supposent ses données : elles ne tournent pas telles quelles ailleurs.

## Limites connues

- **Plateformes vérifiées** : Linux x86_64 (Ubuntu 26.04) ; GPU AMD Radeon 8060S (gfx1151, ROCm 7.1) et NVIDIA GeForce
  RTX 5060 Ti (sm_120, CUDA 13). Autres GPU : non testés ; le F64 de référence et le CPU ne dépendent pas du GPU.
- **Tests GPU hors de la suite par défaut** : environ 115 tests sont ignorés par défaut, dont les tests GPU ; une partie seulement
  est rejouée par la recette interne de l'auteur, et les gates des modules (`rg action-gates`, `rg flux-gates`) n'en font pas partie.
- **Charge topologique de `obs flot` : expérimental, défaut connu.** La charge « trèfle » non améliorée n'est pas quantifiée à
  β = 2,5 (gate GPU `OBS-4-L16` : 5/20 configurations à 0,1 d'un entier après un flot t/a² = 3, seuil 18/20). Ce n'est pas une
  erreur de normalisation (instanton : Q = 1 à 4e-4 ; ⟨Q²⟩ après cooling conforme à la littérature) mais l'effet de
  discrétisation connu de cette définition ; la gate est déclassée, son seuil n'a pas été changé. Pour une charge entière,
  utiliser `kvu obs topo` (charge améliorée). Détail : `docs/DEFAUTS_CONNUS.md`.
- **Régénération** : `kvu regen` rejoue au bit près les manifestes `kvu-regen/1` de `su2 boite` et de `fibre echantillonner`
  (`DIR/manifest_regen.json`) et lit les manifestes historiques `su2_hip/1` et `su2_gen/1` (rejeu sur GPU, binaire admis). Il ne
  rejoue PAS un run `su2 gen` : son `manifest.json` n'est pas reconnu (code 4) et le `regen_bloc3.json` de `--mesures bloc:3`
  échoue au rejeu (code 8) ; défaut PUB-15 de `docs/DEFAUTS_CONNUS.md`. Parade, vérifiée : relancer la commande enregistrée dans
  `manifest.json` (clé `argv`), avec le même binaire et le même backend sur la même machine, dans un dossier `--outdir` neuf et
  sans `--arret-apres` ni `--fenetre-s`, puis comparer les sha256 des fichiers à ceux du manifeste (commandes :
  `crates/kvu/GUIDE.md`, « Régénération des ensembles »).
- **M2** n'estime qu'une action tronquée à quatre boucles et un pas de blocage ; **M4** est CPU F64 et étalonné à L = 4 ;
  **M3** suppose L multiple de 3. Aucun de ces modules ne borne ce qu'il ne mesure pas.
- **Noms hérités** : les profils de machines s'appellent `acemagic` et `gamer` (les deux machines de référence), et les
  variables d'environnement gardent le préfixe `KEVINOTRON_` du moteur interne dont kvu est issu. Les renommer changerait
  l'interface et le format des manifestes : ce sera fait en **0.2**, avec des alias pour les anciens noms.
- **Le fichier `crates/kvu/kvu.toml` est une fixture de test** qui reconnaît les machines de l'auteur par leur GPU (gfx1151,
  sm_120) : lancer `kvu` depuis `crates/kvu/` appliquerait ces profils. Utiliser le modèle de la racine.
- **Compression** : sous un plafond de tâches (`TasksMax`), zstd n'est lancé que sans fil auxiliaire (`--single-thread`, et
  `--no-asyncio` si le zstd installé le connaît, ≥ 1.5.4).
- Documentation et messages en français.

## Historique et nom (History and naming)

- kvu est le successeur nettoyé d'un outillage interne de l'auteur dont le nom de travail était « kevinotron » ; d'où les
  noms `KEVINOTRON_*` (et quelques identifiants internes) qui restent jusqu'à la version 0.2.
- Deux dépôts Zenodo exploratoires de mai 2026 portent ce nom : « Kevinotron v9.0.0 »
  ([10.5281/zenodo.20412608](https://doi.org/10.5281/zenodo.20412608)) et « Crossed Cosmos / Kevinotron v10.0.0 »
  ([10.5281/zenodo.20415354](https://doi.org/10.5281/zenodo.20415354)), versions du même concept 10.5281/zenodo.20379360.
  Ils précèdent kvu ; la version v10.0.0 porte un avis de correction daté du 12 juillet 2026. Leur contenu n'est pas un
  résultat de kvu, et kvu ne le reprend pas.
- kvu v0.1 ne contient aucune affirmation de physique : ce sont des instruments.

*In English.* kvu is the cleaned-up successor of the author's internal tooling, whose working name was "kevinotron" (hence
the `KEVINOTRON_*` names, kept until 0.2). The exploratory Zenodo records of May 2026 bearing that name (10.5281/zenodo.20412608,
10.5281/zenodo.20415354; the latter carries a correction notice dated 12 July 2026) predate kvu and are not results of kvu.
kvu v0.1 makes no physics claim: it provides instruments.

## Logiciel lié, citation, licence

- Générateur SU(2) d'origine (le F64 `compat-prod` de kvu en reproduit les sorties au bit) : **su2-heatbath-gpu**, DOI
  [10.5281/zenodo.23644087](https://doi.org/10.5281/zenodo.23644087).
- Pour citer kvu : `CITATION.cff` ; Zenodo : version 0.1.0 [10.5281/zenodo.24062152](https://doi.org/10.5281/zenodo.24062152), toutes versions [10.5281/zenodo.24062151](https://doi.org/10.5281/zenodo.24062151).
- Licence Apache-2.0 ; dépendances et leurs licences : `LICENCES_TIERCES.md`.
