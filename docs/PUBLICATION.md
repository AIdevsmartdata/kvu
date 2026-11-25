# Préparation de la publication : ce qui a été retiré ou neutralisé (0.1 ; section 0.5.0 en fin de fichier)

Ce dépôt public est un instantané du dépôt de travail de l'auteur. Pour la publication (novembre 2026), trois choses ont
été faites, et seulement celles-ci : des chemins et identifiants privés ont été remplacés par des valeurs neutres, des
dossiers de campagnes internes ont été retirés, et la documentation de tête a été réécrite. Le code de calcul n'a pas
été modifié, à deux exceptions près, signalées dans `CHANGELOG.md` (compression zstd sans fil auxiliaire, repli du
dossier personnel quand `HOME` est absent).

## 1. Neutralisation

Règles appliquées mécaniquement à tous les fichiers texte suivis (fichiers binaires jamais touchés) :

| motif d'origine | remplacé par | nature |
|---|---|---|
| racine du projet de recherche de l'auteur (sous le dossier personnel de l'administrateur) | `/chemin/vers/projet` | chemin privé |
| autres chemins sous le dossier personnel de l'administrateur | `/home/utilisateur/<x>` | chemin privé |
| dossier personnel de l'utilisateur de l'auteur | `/home/utilisateur` | chemin privé |
| deux valeurs fictives d'un test sous `/mnt/…` | `/srv/store`, `/srv/file` | homogénéité du scan |
| nom d'hôte du second poste | `pc-nvidia-exemple` | nom d'hôte |
| adresses IPv4 privées du câble direct entre les deux machines | `192.0.2.<n>` | 192.0.2.0/24 est réservé à la documentation (RFC 5737) |

Conséquences à connaître :

- Les scripts de `crates/kvu/validation/` qui lisaient des données sur les machines de l'auteur pointent maintenant vers
  `/chemin/vers/projet/...` : ils documentent la validation interne et ne tournent pas tels quels ailleurs. Les tests de
  `cargo test` n'en dépendent pas (voir le README).
- Les quelques traces de validation conservées (rendus, avenants, `SHA256SUMS` de fixtures) citent ces chemins neutres.
- Certains fichiers étaient **épinglés** par leur empreinte SHA-256 ailleurs dans l'arbre. Quand l'épingle est contrôlée
  mécaniquement (test cargo, gate `G-H02`, fichier `SHA256SUMS`), elle a été mise à jour. Quand elle appartient à un
  document gelé (pré-enregistrement, protocole de gates), elle a été laissée telle quelle : elle désigne la version
  d'origine. Table de correspondance (16 premiers caractères) :

| fichier | empreinte d'origine | empreinte publiée | épingles laissées à l'origine |
|---|---|---|---|
| `src/hip/su2_sweep.hip` | `da154635a4e92a4d…` | `d04a876a8b5c5e49…` | `crates/kvu/noyaux/su2_commun.h` (commentaire), `ARCHITECTURE_UNIFIEE.md` |
| `crates/kvu/reference/qw1/box_su2.py` | `d7363d6530ad0a91…` | `ffc24bb72f4472f2…` | `GATES-DIR.md` |
| `crates/kvu/validation/hess/outils.py` | `894220f2fa34cbc4…` | `984d7cca4c89626e…` | `prereg/WP14_PREREG.md` |
| `crates/kvu/validation/wp14/fonds.py` | `7d50f11c762b053d…` | `1912a0479c19f689…` | `prereg/WP14_PREREG.md` |
| `crates/kvu/validation/wp14/s2_controle.py` | `6d363fc7c1a639e1…` | `7032e6bf2b5a23ad…` | `prereg/WP14_PREREG.md` |
| `crates/kvu/validation/wp14/verdict.py` | `baeaf7de9abd660d…` | `73b5721d13348119…` | `prereg/WP14_PREREG.md` |
| `crates/kvu/reference/gates_gpu1/analyse_gpu1.py` | `d59426deab1f594c…` | `6126abcfc79ceafa…` | aucune |
| `crates/kvu/reference/gates_gpu1/gates_gpu1.sh` | `d6088a1a49495a26…` | `da3b6a20edf84113…` | aucune |
| `crates/kvu/reference/noyau2/README.md` | `6438e0832920f844…` | `61677a0e383fb9fb…` | aucune |

Dans `src/hip/su2_sweep.hip`, seule la ligne 4 (un commentaire qui donnait le chemin du fichier d'origine) a changé : le
code compilé est identique, et le bloc recopié dans `su2_commun.h` (lignes 27 à 130) aussi, ce que vérifie le test
`su2_commun_recopie_caractere_pour_caractere`. Dans `box_su2.py`, la ligne `#!` désigne maintenant `/usr/bin/env python3`.

## 2. Retiré de l'instantané public

| retiré | raison |
|---|---|
| `rapports/` (audit des ensembles de l'auteur, rapports de régénération) | inventaire des disques et ensembles privés de l'auteur ; aucun test n'en dépend |
| `crates/kvu/validation/wp9_av1/` et `prereg/WP9_AV1_*`, `prereg/WP9_AVENANT_AV1.md`, `prereg/WP9_RECALCUL_AV1_*` | campagne interne de ré-analyse (données, résultats gelés, pré-enregistrements) dont les empreintes sont chaînées : neutraliser les chemins aurait cassé la chaîne ; aucun test cargo n'en dépend |
| journaux et résultats de validation interne (`crates/kvu/validation/**/resultats*` : banc, f32, fibre_contrainte, historique, integration ; 89 fichiers) | traces des campagnes internes sur les machines de l'auteur ; aucun test cargo n'en dépend ; les 3 tests Python de `fibre_contrainte/test_reverdict_newton.py` qui les lisaient sont sautés quand elles manquent |
| `kvu.toml` racine, version d'origine (lien vers les profils des machines de l'auteur) | remplacé par un **modèle** neutre ; les profils d'origine (neutralisés) sont la fixture `crates/kvu/kvu.toml` |

Rien d'autre n'a été retiré. Les fichiers retirés restent dans le dépôt de travail de l'auteur.

## 3. Ce qui n'a pas été changé

- Les noms de profils de machines `acemagic` et `gamer`, et le mot `kevinotron` (ancien nom du moteur), restent dans le
  code, les tests et les variables d'environnement (`KEVINOTRON_*`) jusqu'à la 0.2 (décision de l'auteur) : les renommer
  change l'interface et le format des manifestes. Ils ne désignent aucune donnée privée.
- Les messages de commit et l'historique du dépôt de travail ne font pas partie de l'instantané.

## 4. Version 0.5.0 (novembre 2026)

Même démarche que pour la 0.1, appliquée à la ligne de développement (`integration/v0.5`, qui contient les versions internes 0.2 à 0.4) :

- Retirés, comme pour la 0.1 : `rapports/`, `crates/kvu/validation/wp9_av1/`, les trois pré-enregistrements WP9-AV1, les journaux et résultats de
  validation interne (`crates/kvu/validation/**/resultats*`, dont 122 fichiers apparus depuis la 0.1). Retirés en plus : huit scripts de campagne
  qui portent des chemins absolus de la machine de l'auteur (`scripts/accord_xi.sh`, `accord_xi.py`, `balayage_xi2.sh`, `analyse_point_xi.py`,
  `test_aniso_bits.sh`, `test_aniso_stat.sh`, `test_ops_bruts.sh`, `verif_ops_bruts.py`).
- Neutralisés à la main : `scripts/lecture_graine/*` (chemins de données de production remplacés par `/chemin/vers/…` ou la variable
  `KVU_DONNEES_REELLES` ; ces tests exigent des données de l'auteur et ne tournent pas telles quelles), `docs/ETAT_v0.3.md`, le fichier de test
  `crates/kvu/tests/donnees/reprise_0_1_1/manifest.json` (chemins de travail remplacés par `/chemin/travail/…`).
- Remplacés par la version publiée de la 0.1.1 (la ligne de développement portait l'état d'avant) : `docs/DEFAUTS_CONNUS.md` (complété d'un avertissement),
  `crates/kvu/validation/machine/test_build_auto*.sh`. Rétablis : `docs/NOTE_0.1.1.md`, `scripts/verifier_mesures_en_ligne.py`.
- Non faits : ajout d'en-têtes SPDX aux fichiers source (la licence est déclarée par `LICENSE`, `NOTICE`, `Cargo.toml` et `CITATION.cff` ; modifier les
  sources changerait leur empreinte `kvu-source/1` et les épingles des tests) ; renommage des noms hérités. Les deux sont reportés à la 0.6 (le fichier `LICENSE` Apache-2.0 et le champ `license` de `Cargo.toml` portent la licence en 0.5.1).
- Recette (tests verts, clippy rouge : 65 lints de style non corrigés) : voir `CHANGELOG.md` (« Recette de cette version »). Le code Rust et les noyaux sont identiques, octet pour octet, à ceux de la tête `e9f9f12` de
  la ligne de développement (même empreinte `kvu-source/1` : `ec75d990…`) ; les qualifications de carte graphique de cette ligne (identité au bit AMD et NVIDIA,
  tests de carte) datent du 23 novembre au matin et n'ont PAS été rejouées depuis l'archive publique, à l'exception de deux essais courts sur la carte AMD
  (voir les notes de version).
- Aucun changement de code Rust ni de noyau. Les seuls fichiers de test ou de fixture modifiés sont ceux cités plus haut (`test_build_auto*.sh`, `reprise_0_1_1/manifest.json`, `scripts/lecture_graine/test_*.py`).

## 5. Version 0.5.1 (23 novembre 2026)

La 0.5.0 était une version interne, non publiée ; la 0.5.1 est la version publique de la ligne 0.5 : 0.5.0 interne + reports de correctifs de reprise + durabilité avant la clôture + gardes d'entrée + lints (détail : `CHANGELOG.md`).

- Reportés depuis la 0.1.1 : PUB-14 et PUB-16 (trois commits, `crates/kvu/src/su2/run.rs`, tests `pub14_*` et `pub16_*` dans `crates/kvu/tests/su2_reprise_mesures.rs`), puis, après un audit des correctifs de la 0.1.1 un par un : PUB-14 pour `--bord` (`su2/bord.rs`) et `--mesures ""` à la reprise (`su2/run.rs`), avec les tests `pub01_*`, `pub02_bord_et_mesures_refuse`, `run_sans_mesures_sans_compagnon`, `mesures_vide_vaut_sans_mesures` et `pub14_bord_kill9_en_thermalisation_reprise_acceptee` de la 0.1.1 (détail : `docs/DEFAUTS_CONNUS.md`, note de tête).
- Durabilité avant la clôture (chemin standard) et gardes d'entrée : fichiers consignés synchronisés avant `manifest.json` et `COMPLETE` ; synchronisation du répertoire après les renommages de `reprise.champs` avant `reprise.json`, erreur de synchronisation rendue par l'écriture atomique ; remise en place de la génération désignée avant chaque point (deux arrêts successifs) ; volumes de `su2 thermo mesure` contrôlés (`checked_mul`) ; `etat.bin` : `usize::try_from` (nombre de liens et identifiant de balayage) ; garde `b >= usize::BITS` dans `gpu_torelon.rs`. La reprise après coupure de courant n'est pas établie (PUB-05, PUB-20). Aucune boucle numérique modifiée.
- `cargo clippy --locked -p kvu --all-targets -- -D warnings` vert (avec et sans `--features cuda`) : corrections de style (`is_multiple_of`, `div_ceil`, `slice::from_ref`, un `0 *` inutile, deux branches identiques fusionnées) et `#[allow(clippy::needless_range_loop | too_many_arguments | type_complexity)]` locaux commentés sur les fonctions concernées, `#![allow(clippy::doc_lazy_continuation | doc_overindented_list_items)]` de fichier pour des listes de documentation. Aucune boucle numérique n'a été réécrite.
- Trois mentions d'un projet sans rapport retirées (`machine/processus.rs`, `validation/machine/kevinotron-queue.service.exemple`, `reference/noyau2/README.md` ; l'empreinte de ce dernier fichier est mise à jour dans `validation/historique/SHA256_REFERENCE.txt`, test G-H02).
- Identité numérique : `outils/identite_au_bit.sh` (script du paquet de préparation, non publié ; il exige le binaire interne de référence) `BINAIRE_0.5.0_e9f9f12 BINAIRE_0.5.1 SORTIE` lance sept essais processeur (`su2 gen` L = 6 avec mesures wilson et flot ; archive zstd ; torsion ; réseau anisotrope `--cotes --xi0` ; points de reprise ; couture décalée ; `su2 spectro neutre`) avec les deux binaires et compare tous les fichiers de sortie, SAUF les manifestes, `COMPLETE`, les journaux, `config_run.json`, `bilan*` et `regen*` (exclus) et, pour `archive.json` et `resume.json`, hors champs de provenance, chemins et durées. Résultat : **7 essais sur 7 identiques au bit** (`preuves/identite_au_bit.txt`). Binaire de référence : la qualification de ce jour (`kvu 0.5.0`, source `ec75d990…`, git `e9f9f12`). Éléments exclus de la comparaison : manifestes (`manifest*`), `COMPLETE`, journaux (`*.log`), `config_run.json`, `bilan*`, `regen*`, et, pour `archive.json` et `resume.json`, les champs de provenance (binaire, empreintes, version, chemins) et les durées ; tout le reste de chaque dossier de sortie est comparé octet pour octet.
  Liste exhaustive des sept essais (chaque ligne : nom, puis arguments de `kvu` ; `@OUT` = dossier de sortie) :

```
  cas ens_L6 su2 gen --backend cpu --L 6 --beta 2.3 --graine 1 --start cold --therm 20 --nmeas 20 --nor 1 --dump-every 10 --format q64 --mesures wilson:R2xT2:1,flot:0.02:0.2:10 --outdir @OUT
  cas arch_L4 su2 gen --backend cpu --L 4 --beta 2.3 --graine 2 --start cold --therm 10 --nmeas 10 --nor 1 --dump-every 5 --format q64 --zstd --archive --outdir @OUT
  cas tordu_L4 su2 gen --backend cpu --L 4 --beta 2.3 --graine 3 --start cold --therm 10 --nmeas 10 --nor 1 --torsion 01,23 --outdir @OUT
  cas aniso su2 gen --backend cpu --cotes 4,4,4,6 --xi0 1.5 --beta 2.3 --graine 5 --start hot --therm 10 --nmeas 10 --nor 1 --dump-every 5 --outdir @OUT
  cas reprise_pt su2 gen --backend cpu --L 4 --beta 2.3 --graine 7 --start cold --therm 12 --nmeas 12 --nor 1 --reprise-every 4 --dump-every 6 --format q64 --mesures wilson:R1xT1:1 --outdir @OUT
  cas spectro_neutre su2 spectro neutre --cotes 4,4,4,8 --beta 2.3 --graine 11 --therm 10 --nmeas 10 --skip 2 --nor 2 --sortie @OUT
  cas decalage su2 gen --backend cpu --cotes 4,4,4,4 --decalage 0,1,0,0 --beta 2.3 --graine 6 --start cold --therm 10 --nmeas 10 --nor 1 --dump-every 5 --outdir @OUT
```

- Les empreintes de source (`kvu-source/1`) de la 0.5.1 et de la 0.5.0 diffèrent (commentaires, correctifs, version) : les manifestes des mesures de ce jour portent « 0.5.0 » et l'empreinte `ec75d990…`.
- Essais sur la Radeon (HIP) et état de la validation sur carte : section « État de la validation sur carte (0.5.1) » du `README.md` (bilan détaillé de référence).
