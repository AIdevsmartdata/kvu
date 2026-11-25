# Champs par site pour le scanner : `kvu obs champs` (format `kvu-champs/1`, 11/11/2026)

Statut : outil d'observation. Une image tirée de ces champs montre UNE configuration ; aucune image ne prouve un trou de masse.

## Commande
```
kvu obs champs --cfg X.npy --out DIR [--t-flot 0.5 | --n-pas 50] [--dt 0.01] [--beta 2.3] [--dir-temps 3] [--f64] [--porte-jauge] [--graine-jauge 1] [--sans-portes]
```
**Refus** : une configuration tordue (torsion de 't Hooft) ou non périodique (boîte, bord, masque) est refusée AVANT tout
calcul, rien n'est créé : les densités ignorent les facteurs z_p = −1 de la pile de torsion. La déclaration est lue dans
`torsion.json`, `manifest.json` (`torsion`, `bord`, `masque`, `mode` su2-bord/su2-boite, `config_run`) et `config_run.json`, dans le
dossier de la configuration et son parent. Sans aucune de ces métadonnées, la configuration est SUPPOSÉE périodique, et
`geometrie.declaration` le dit. Le chemin de `--cfg` est d'abord rendu ABSOLU et résolu (liens symboliques compris) : `c000/cfg.npy`
depuis la racine d'un lot, `cfg.npy` depuis `c000/`, `../c000/cfg.npy` ou un lien cherchent aux mêmes dossiers (celui de la cible
et son parent) ; un chemin qu'on ne sait pas résoudre est une erreur, jamais ignoré.
**Autres refus** (code 2, usage, avant toute création) : `--t-flot` avec un `--dt` qui donne plus de 2^53 pas (par exemple
`--t-flot 1 --dt 1e-30`), un temps de flot n_pas·dt non fini. **Champ non fini** : si un des sept champs (ou ceux de la
transformation de jauge) contient un NaN ou un infini (par exemple `--dt` énorme), l'export est refusé (code 8, validation),
AVEC ou SANS `--sans-portes`, et rien n'est marqué COMPLETE.
Entrée : une configuration SU(2) lisible par `stockage::npy::lire_champ` (NPY c16 `(4,L,L,L,L,2,2)` complex128, ou q64, `.zst`
compris). Calcul sur le CPU (`obs::ref_cpu`, FP64, rayon) : le manifeste consigne `backend = "cpu"`, `arch = null`, `precision_calcul = "FP64"`
quelle que soit la machine. Coût mesuré (Vast, 24 fils) : L = 24, 50 pas de flot, portes de jauge
comprises : 29 s, 185 Mo de mémoire résidente.

## Sorties
`DIR` doit être NEUF. `DIR/<champ>.npy` (NPY v1.0, petit-boutiste, ordre C ; `<f4` par défaut, `<f8` avec `--f64`) et le
manifeste COMMUN de kvu `DIR/manifest.json` (`stockage::provenance::Manifeste` : provenance unifiée, `fichiers` avec sha256 et
octets, `statut`, `rc`, `COMPLETE` seulement si toutes les portes sont vertes).

Disposition : un champ 4D a la forme `(L, L, L, L)`, indexée `[x0, x1, x2, x3]`, c'est-à-dire le site kvu
`s = ((x0·L + x1)·L + x2)·L + x3`. `polyakov` a la forme `(4, L, L, L, L)`, indexée `[d, x0, x1, x2, x3]`.
La plaquette « avant » de coin x dans le plan (μ,ν) est `U_μ(x) U_ν(x+μ) U_μ(x+ν)† U_ν(x)†` ; plans dans l'ordre 01 02 03 12 13 23.

| champ | définition | agrégat = observable globale de kvu |
|---|---|---|
| `action_brute` | e(x) = 4 Σ_{μ<ν} (1 − ½ Tr P_μν(x)) | moyenne = E_plaq = 24(1 − ⟨P⟩) (`ref_cpu::densites`) ; S_W = (β/4) Σ_x e(x) |
| `action_flot` | e(x) après `n_pas` pas du flot de Wilson RK3 (`ref_cpu::flot_n_pas`), t = n_pas·dt | moyenne = E_plaq(t) (`ref_cpu::flot`) |
| `energie_trefle_flot` | E_trèfle(x) = 2 Σ_plans |g_μν(x)|², G = (Q − Q†)/8, Q = trèfle de 4 plaquettes | moyenne = E_trèfle(t) |
| `charge_flot` | q(x) = (1/2π²)(g01·g23 − g02·g13 + g03·g12), trèfle, après flot | somme = Q_top(t) |
| `polyakov` | ℓ_d(x) = ½ Tr ∏_k U_d(x + k ê_d), configuration BRUTE ; constant le long de d ; réel pour SU(2) ; norme = |ℓ_d(x)| | moyenne = ⟨P_d⟩ (`ref_cpu::polyakov_moyen`) |
| `plaq_negatives_brute` | nombre (0-6) de plaquettes avant de coin x avec ½ Tr P < 0 | somme = compte par `ref_cpu::boucle(·,1,1)` |
| `plaq_negatives_flot` | idem après flot | idem |

`plaq_negatives_*` est le **signe de la trace** des plaquettes : invariant de jauge, il n'utilise ni jauge maximale du centre ni
projection de centre ; ce ne sont pas des P-vortex.

Champs propres du manifeste : `geometrie` {l, dimensions, axes, conditions_aux_bords, torsion, direction_temps (`--dir-temps`,
consignée seulement), declaration} et `kvu_champs` (format `kvu-champs/1`) : `l`, `beta` (donné par `--beta`, sinon null),
`disposition`, `flot` {méthode, dt, n_pas, t, sqrt8t}, `source` {chemin, sha256}, `kvu` {version, sha256 du binaire}, `champs`
{nom → fichier, forme, dtype, sha256, définition, agrégat comparé, somme et moyenne FP64 (avant arrondi f32), min, max},
`references_kvu`, `porte_jauge` {graine, transformation}, `portes` (une entrée par porte), `portes_ok`, durées.

## Portes (seuil 1e-10 ; PREREG du chantier `kvu_modules_2026-11-11/scanner/PREREG.md`)
- CH-SOMME-* : chaque agrégat FP64 contre l'observable globale recalculée par une entrée indépendante de kvu (`densites`,
  `table_wilson` W(1,1), `flot`, `polyakov_moyen`, compte par `boucle`). Échelle de l'écart relatif : |référence| pour E ; Σ|q(x)|
  pour Q et ⟨|ℓ_d|⟩ pour Polyakov (grandeurs signées dont la somme peut s'annuler) ; égalité exacte pour les comptes.
- CH-LOCALITE : 64 sites, e(x) et ℓ_d(x) recalculés par des chemins indépendants (une permutation des sites garde les sommes, pas
  cette porte).
- CH-JAUGE-<champ> (`--porte-jauge`) : max|f − f_g| ≤ 1e-10 · max(max|f|, 1) après une transformation de Haar aléatoire de tous
  les sites (plancher O(1) du dénominateur : relatif pour un champ d'échelle ≥ 1, absolu pour un champ nul ou presque nul, où les
  arrondis de jauge ~1e-14 n'ont pas de sens relatif ; limite : une non-invariance < 1e-10 sur un champ < 1 n'est pas vue).
- Toutes les portes de maximum propagent les NaN (`f64::max` les ignorerait) : un champ non fini rend la porte rouge.
- Mutants dans `crates/kvu/tests/obs_champs.rs` (plaquette oubliée, charge de signe opposé, pas de flot en trop, site en trop,
  Polyakov ×1,001, champ translaté, champ non invariant, champs NaN et infinis) : tous rouges. Les essais de commande tournent en
sous-processus isolé (seuil SSD de test explicite), comme `tests/obs_topo_cli.rs`.

## Lire en Python
```python
import json, numpy as np
m = json.load(open("DIR/manifest.json"))["kvu_champs"]
q = np.load("DIR/charge_flot.npy")          # (L,L,L,L), q[x0,x1,x2,x3]
assert abs(q.astype(np.float64).sum() - m["champs"]["charge_flot"]["somme_f64"]) < 1e-5 * np.abs(q).sum()
```
