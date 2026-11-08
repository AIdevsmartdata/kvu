# Préparation de la publication v0.1 : ce qui a été retiré ou neutralisé

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
