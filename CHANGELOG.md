# Journal des versions

Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) ; versions selon [SemVer](https://semver.org/lang/fr/).
Tant que la version majeure est 0, l'interface (options, formats de manifeste, variables d'environnement) peut changer
d'une version mineure à l'autre ; chaque changement est noté ici.

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
PUB-18 et PUB-19 sont deux limites aux marges de la chaîne de Markov, hors de tout usage connu : plus de 2³² balayages dans un
même run (le compteur du générateur aléatoire repasse à zéro) ; β extrêmement petit en F64 (le bain thermique ne tire plus rien).
PUB-15 : `kvu regen` ne rejoue pas un run `su2 gen` (ni son `regen_bloc3.json`) ; le README et le GUIDE ne le promettent plus
et donnent la parade (relancer l'`argv` du manifeste, comparer les sha256).

## [0.1.0] — 2026-11 (première version publique)

Premier instantané public du dépôt de travail (`main` 9362d77 du 2 novembre 2026, plus les changements de publication
ci-dessous).

### Contenu

- Générateur SU(2) Wilson 4D `su2 gen` : bain thermique + surrelaxation, Philox4x32-10 à compteur, F64 de référence et F32
  rapide, source unique de noyaux pour HIP et CUDA, référence CPU F64, lots de chaînes, reprise au bit, mesures en ligne,
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
  des gates historiques ont leur propre store et ne dépendent plus de la place libre de la machine (`KVU_SSD_MIN_GO`) ;
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
  pas quantifiée à β = 2,5, L = 16, t/a² = 3 (5/20 contre ≥ 18/20 exigés, sur CUDA comme sur HIP ; rejeu identique au bit
  sur CUDA). Ni
  normalisation ni flot en cause ; effet de discrétisation connu (Athenodorou et Teper, arXiv:2106.00364). Observable marqué
  expérimental ; `obs topo` donne la charge améliorée. Test et seuil inchangés. Voir `docs/DEFAUTS_CONNUS.md`.

### Retiré de l'instantané public

- `rapports/`, la campagne interne WP9-AV1 (`crates/kvu/validation/wp9_av1/`, pré-enregistrements associés) et les journaux
  de validation interne (`crates/kvu/validation/**/resultats*`) : voir `docs/PUBLICATION.md`.

### Historique et nom

- kvu succède à l'outillage interne « kevinotron » de l'auteur ; les dépôts Zenodo exploratoires de mai 2026 qui portent ce nom
  (10.5281/zenodo.20412608, 10.5281/zenodo.20415354) précèdent kvu et n'en sont pas des résultats (voir le README).
- Prévu pour **0.2** : renommage des variables `KEVINOTRON_*`, des défauts `~/kevinotron_store` et `/tmp/kevinotron_gpu.lock`
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
| `cargo clippy --all-targets -- -D warnings` | rc 0 | 10 s |
| `cargo test --features cuda --no-run` | rc 0 (le backend CUDA compile ; rien n'est lancé) | 43 s |
| `scripts/exemples_readme.sh` | les 11 commandes du README rendent 0, sorties attendues présentes | 2 s |
| tests Python de la recette (`scripts/tests/`) | 7/7 | < 1 s |

La somme des lignes « test result » de cargo donne 1468 « passed » : cinq tests de `tests/su2_cpu.rs` relancent leur
propre exécutable dans un processus enfant, dont le résumé s'ajoute. Le décompte exact est celui de la liste (1463 actifs).
Aucun dossier n'a été créé hors de l'arbre de travail et du dossier temporaire. Les tests GPU n'ont pas été lancés pour
cette recette.
