# Archivage vérifiable des configurations `kvu su2 gen`

## But

`kvu su2 gen --archive` produit, en plus des dumps compressés, un index `archive.json`
(schéma `kvu-archive/1`) qui relie chaque configuration à son indice de mesure, à ses
balayages (`therm + m·skip`, ou 0 pour le dump initial), aux empreintes SHA-256 du
fichier compressé ET du contenu décompressé, et à la provenance du run.

Le lecteur (`kvu su2 archive extraire`) vérifie ces empreintes avant de rendre un champ ;
le vérificateur (`kvu su2 archive verifier`) liste tous les défauts sans s'arrêter au premier.

## Garanties

- **Intégrité du fichier compressé** : taille et SHA-256 du `.zst`.
- **Intégrité du contenu décompressé** : SHA-256 et taille du `.npy` décompressé en flux,
  après vérification du code de retour de `zstd`.
- **Lien indice ↔ balayages** : l'index stocke `balayages` ; toute incohérence est un défaut
  `index` (ou `champ` pour un contenu illisible). Le produit et la somme sont contrôlés :
  un débordement est refusé, même si `balayages` correspond au résultat modulo 2⁶⁴.
- **Lien fichier ↔ indice** : seul un nom canonique `cfg_NNNNNN.npy.zst` (6 chiffres exacts)
  dont l'indice décodé égale l'indice de l'entrée est accepté à la lecture comme à la construction.
  Les chemins absolus, `..` et séparateurs sont refusés.
- **Provenance** : sept champs (`binaire`, `binaire_sha256`, `argv`, `source_hash`, `git_head`,
  `version`, `hote`) copiés depuis `manifest.json` ; `null` si absent.
- **Cohérence avec `config_run.json`** : L, nombre de chaînes, β et graines sont comparés.
  La structure est validée par désérialisation en `ConfigRun` ; `dump_initial` et `zstd`
  sont des booléens obligatoires. Une configuration présente mais invalide produit un
  défaut `index` (rc 8 au vérificateur).

## Non-garanties

- Le SHA-256 du fichier compressé dépend de la version de `zstd` : comparer des archives
  entre machines par `sha256_contenu`, pas par `sha256`.
- L'archive ne contient pas les CSV ni le manifeste : elle ne remplace pas la sauvegarde complète.
- Un fichier ajouté avec un nom `cfg_*` non indexé est signalé `non_indexe`, mais l'indexation
  ne le supprime pas.

## Schéma

```json
{
  "format": "kvu-archive/1",
  "l": 4,
  "nor": 2,
  "precision": "f64",
  "therm": 10,
  "nmeas": 20,
  "skip": 2,
  "dump_every": 5,
  "dump_initial": false,
  "format_champ": "q64",
  "compression": "zstd",
  "provenance": { "...": "copié de manifest.json" },
  "complet": true,
  "chaines": [
    {
      "chaine": 0,
      "dossier": ".",
      "beta": 2.3,
      "graine": 26110201,
      "depart": "chaud",
      "configurations": [
        {
          "fichier": "cfg_000010.npy.zst",
          "indice": 10,
          "balayages": 30,
          "octets": 1234,
          "sha256": "…",
          "octets_contenu": 2654336,
          "sha256_contenu": "…"
        }
      ]
    }
  ],
  "note": "index reconstruit depuis les fichiers ; vérifier avec kvu su2 archive verifier"
}
```

## Commandes

```text
kvu su2 gen … --archive                         # active l'archivage (exige --dump-every N ≥ 1 et zstd)
kvu su2 archive verifier DIR                    # rc 0 si aucun défaut, 8 sinon
kvu su2 archive indexer DIR                     # (re)construit archive.json d'un run existant compressé
kvu su2 archive extraire DIR --indice M [--chaine C] --sortie F.npy [--format c16|q64]
kvu su2 archive lister DIR                      # une ligne par configuration
```

`extraire` et `lister` n'exigent pas `config_run.json` ; `indexer` et `verifier` l'exigent
(rc 2 si absent). Un indice absent de l'index est une erreur d'usage (rc 2) ; un fichier non
indexé présent sur le disque est signalé `non_indexe` par le vérificateur.

Sans `--chaine`, `extraire` sélectionne la chaîne 0 ; `--chaine 1` sélectionne `c001`
dans un lot. Sans `--format`, le fichier extrait conserve le format déclaré par l'archive
(`q64` ou `c16`).

## Reprise

Avec `--archive`, chaque arrêt (terminé, `--arret-apres`, `--fenetre-s`) reconstruit l'index
depuis les fichiers présents. `--reprendre DIR` active le mode archive si `archive.json` existe
ou si `--archive` est redonné. Un premier passage tué avant la fin n'a pas d'index : redonner
`--archive`.

## Recommandation

Utiliser `--format q64` (2× plus petit que c16) et laisser `--zstd` implicite avec `--archive`.

## Exemple

```console
$ kvu su2 gen --backend cpu --L 4 --beta 2.3 --graine 26110201 --start hot \
  --therm 10 --nmeas 20 --skip 2 --nor 2 --dump-every 5 --format q64 --archive --outdir run
su2 gen : archive.json : 4 configurations indexées

$ kvu su2 archive verifier run
(rien, rc 0)

$ kvu su2 archive lister run
0 5 20 cfg_000005.npy.zst aaaa…
0 10 30 cfg_000010.npy.zst bbbb…
0 15 40 cfg_000015.npy.zst cccc…
0 20 50 cfg_000020.npy.zst dddd…

$ kvu su2 archive extraire run --indice 10 --sortie cfg_000010.npy
run/cfg_000010.npy.zst : configuration 10 extraite
