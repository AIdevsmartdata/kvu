# Bord arbitraire d'une boîte lu dans un fichier NPY (`kvu su2 boite --bord-npy`)

Lot WP21 du programme `kvu` (tranche 4), spécification écrite AVANT le code (27/10/2026). Besoin : la campagne DESERT_GPU (chantier vision)
veut optimiser les bords (glouton, recuit) ; `su2 boite` n'acceptait que 7 bords nommés. Voir `docs/BOITES.md` pour les boîtes elles-mêmes.

## 1. Ligne de commande

```text
kvu su2 boite --ell 8 --beta 2.5 --bord-npy BORD.npy --chaines npy:1a2b3c4d:26102700001,cold:26102700002 --therm 1000 --nmeas 30000 --outdir DIR
kvu su2 boite --ell 8 --beta 2.5 --bord-npy BORD.npy --bords npy --repliques 16 --graine 26102700100 --outdir DIR2      # un seul fichier : « npy »
kvu su2 boite --ell 8 --beta 2.5 --bord-npy A.npy --bord-npy B.npy --chaines npy:1a2b3c4d:1,npy:9f8e7d6c:2 --outdir DIR3   # plusieurs fichiers
```

- `--bord-npy FICHIER` est **répétable** : chaque occurrence ajoute un bord au lot.
- **Nom du bord** : `npy:<h8>`, où `<h8>` = les 8 premiers caractères hexadécimaux (minuscules) du sha256 du fichier (`sha256sum FICHIER | cut -c1-8`).
  Il s'emploie comme un nom de bord ordinaire, dans `--chaines npy:<h8>:GRAINE,…` et dans `--bords cold,npy:<h8>,…`.
- **Raccourci** : quand UN SEUL `--bord-npy` est donné, le nom `npy` (sans suffixe) le désigne. Il est remplacé par `npy:<h8>` dans la configuration,
  le manifeste et la commande canonique.
- `--chaines` découpe désormais chaque élément au **dernier** « : » (`npy:1a2b3c4d:7` ⇒ bord `npy:1a2b3c4d`, graine 7) ; les noms existants ne
  contiennent pas de « : », leur lecture est inchangée.
- **Graine** : comme pour tout bord, la graine d'une chaîne est la clé Philox de son bain thermique ; un bord lu ne dépend d'aucune graine.
  Conséquence : un fichier qui contient exactement le bord nommé `b` de la chaîne `b:G` redonne, avec la graine G, la chaîne `b:G` à l'octet
  (gate BORD-NPY-EQUIV).
- Un lot peut mêler bords nommés et bords lus ; plusieurs chaînes peuvent partager un même fichier.

## 2. Format du fichier

Le MÊME format que `--dump-final` (q64) et que `su2 boite-obs --cfg` (q64 ou c16) : un champ de liens du **réseau de calcul** de la boîte.

| élément | exigence |
|---|---|
| conteneur | NPY v1.x (signature `\x93NUMPY`, version majeure 1), non compressé (un `.npy.zst` est refusé) |
| ordre | `fortran_order: False` (C-order) |
| q64 | `descr '<f8'`, forme `(4, Lc, Lc, Lc, Lc, 4)` : quaternion `(a0, a1, a2, a3)`, scalaire en premier, U = a0 + i(a1σ1 + a2σ2 + a3σ3) |
| c16 | `descr '<c16'`, forme `(4, Lc, Lc, Lc, Lc, 2, 2)` : matrice 2×2 complexe, U = [[a0 + i a3, a2 + i a1], [−a2 + i a1, a0 − i a3]] |
| côté | **Lc = ℓ + 2** (réseau de calcul de la boîte, celui de `--dump-final`) |
| taille | exactement en-tête + 4·Lc⁴·(32 octets en q64, 64 en c16) : ni tronqué, ni octets en trop |
| indice | élément `[μ, x0, x1, x2, x3]` = lien U_μ(x) de x à x + e_μ (μ = 0…3) ; indice plat `li = μ·Lc⁴ + ((x0·Lc + x1)·Lc + x2)·Lc + x3` |

**Liens lus.** Seuls les liens **fixés** de la boîte sont lus : x ∈ {0,…,ℓ}⁴, x_μ ≤ ℓ − 1, et x comme x + e_μ sur le bord du cube (une coordonnée
vaut 0 ou ℓ) ; il y en a `n_fixes(ℓ) = 4ℓ(ℓ+1)³ − 4ℓ(ℓ−1)³` (ℓ = 6 : 5 232 ; ℓ = 8 : 12 352). Les autres éléments du fichier (liens libres, liens
inexistants, couche de remplissage) ne sont **pas décodés** : leur contenu, quel qu'il soit (même NaN), n'a aucun effet sur la sortie ; seule
l'empreinte sha256 du fichier en dépend. Le départ intérieur reste froid (identité), comme pour les bords nommés.

**Liens fixés actifs et inertes** (constat du lot WP21, 28/10). Seuls les liens fixés qui appartiennent à une plaquette de la boîte
contenant un lien libre entrent dans la dynamique (agrafes des liens libres) : ℓ = 4 : 864 sur 1 568 ; ℓ = 6 : 3 600 sur 5 232 ; ℓ = 8 : 9 408
sur 12 352 ; ℓ = 12 : 34 848 sur 41 568 (soit 24ℓ(ℓ−1)² pour ces ℓ ; dénombrement exhaustif, script `scratch/wp21/inertes.py` du lot). Les autres
(arêtes et coins du bord) sont **inertes** : les modifier ne change ni la trajectoire ni les séries centrales, seulement les plaquettes de bord
du profil `d0` et la configuration finale (gate BORD-NPY-MUT). Une optimisation de bord (glouton, recuit) peut se restreindre aux liens actifs.

**Validation des liens fixés** (même règle que `lire_champ`, donc que `boite-obs --cfg`) : q64 : les 4 réels finis et `| |q| − 1 | ≤ 1e-12` ;
c16 : `q = mat_to_quat(M)`, les 4 réels de q finis, et `max( max|M − quat_to_mat(q)|, | |q| − 1 | ) ≤ 1e-12` (structure SU(2) exacte).
Un lien fixé hors tolérance, NaN ou ±Inf ⇒ refus, avec son indice `li` et (μ, x).

**Valeurs copiées au bit.** Aucune renormalisation : les 4 réels de chaque lien fixé (q64), ou les 4 demi-sommes exactes de `mat_to_quat` (c16 ;
exactes sur une matrice écrite par `quat_to_mat`), sont posés tels quels dans le champ de départ. Un fichier écrit par `--dump-final` d'un run au
bord `b` contient donc exactement le bord `b` (contrôle `liens_geles_intacts` de ce run).

## 3. Refus (code de sortie ≠ 0, message explicite ; rien n'est lancé et aucun répertoire de sortie n'est créé)

| cas | code |
|---|---|
| fichier absent ou illisible | 4 (E/S) |
| pas un NPY v1 (signature, version), en-tête illisible, `fortran_order` vrai, dtype ou forme autres que q64 / c16, taille ≠ en-tête + données | 4 (format) |
| côté Lc ≠ ℓ + 2 | 2 (usage) |
| lien fixé hors SU(2) à 1e-12, NaN ou ±Inf | 4 (format) |
| chaîne `npy…` sans aucun `--bord-npy` ; `npy:<h8>` qu'aucun fichier ne porte ; `npy:` suivi d'autre chose que 8 hexadécimaux minuscules | 2 |
| `npy` nu avec deux fichiers ou plus | 2 |
| un `--bord-npy` qu'aucune chaîne n'utilise | 2 |
| deux `--bord-npy` de même nom `npy:<h8>` (même contenu, ou collision des 8 premiers hexadécimaux) | 2 |
| `--bords` et `--chaines` ensemble, ni l'un ni l'autre, et tous les refus existants | 2 (inchangés) |

## 4. Manifestes

- `manifest.json` (`kevinotron-unifie/1`) : champ `bords_npy`, présent seulement s'il y a au moins un fichier, liste dans l'ordre des options :
  `{"nom": "npy:<h8>", "chemin": "<chemin absolu canonique>", "sha256": "<64 hex>", "octets": N, "format": "q64"|"c16", "l_calcul": Lc,
  "n_liens_fixes": n}`. Les chaînes de `config_boite` portent le nom `npy:<h8>`.
- `manifest_regen.json` (`kvu-regen/1`) : `entrees` = les mêmes entrées, avec en plus `"role": "bord-npy"` (le schéma exige `nom` et `sha256`) ;
  `parametres.boite.bords_npy` = la même liste ; `commande` = `su2 boite --ell … --beta … --bord-npy <chemin absolu> [--bord-npy …] --chaines
  npy:<h8>:G,… …` (les options `--bord-npy` juste avant `--chaines`, dans l'ordre d'origine). Sans fichier, `entrees` reste `[]` et la commande
  canonique est inchangée.

## 5. Rejeu (`kvu regen`)

Avant tout rejeu (avant l'admission du binaire et avant de créer quoi que ce soit), chaque entrée de `entrees` qui porte un `chemin` est hachée :
fichier absent ⇒ refus code 4 ; sha256 ≠ manifeste ⇒ refus code 8 (« entrée modifiée »), rien n'est rejoué. Le rejeu exige donc le fichier au
même chemin absolu. Les manifestes existants (`entrees` vide) ne sont pas concernés.

## 6. Mémoire et limites

- Les liens fixés de chaque fichier distinct restent en mémoire hôte (32 octets par lien fixé ; ℓ = 16 : 3,1 Mo par fichier), comptés dans
  l'admission mémoire du lot ; le fichier est lu en flux (jamais entier en mémoire).
- Le pilote `scripts/pilote_boites.py` ne gère pas les bords lus (pas de colonne de fichier) : lancer `kvu su2 boite` directement.
- Un fichier n'est pas recopié dans le répertoire de sortie : il doit être conservé (au même chemin) pour le rejeu.
