# BANC_UNIFIE — kevinotron unifié, bancs des deux machines

Produit par `validation/banc/rapport_banc.py` à partir des JSON de `kevinotron bench` (verrou GPU, fenêtres ≤ 600 s,
GPU rendu entre deux fenêtres). Rapporté, non gaté, sauf la gate de non-régression : **F64 unifié ≥ 0,9 × su2_hip / su2_cuda**
(référence lancée dans la même fenêtre, mêmes L, β = 2,5, 4 OR par pas, une chaîne).

## Gate de non-régression F64 : **FAIL**

| machine | L | référence | pas/s unifié ÷ référence | gate (≥ 0,9) |
|---|---|---|---|---|
| acemagic | 8 | su2_hip | 0.767 | FAIL |
| acemagic | 16 | su2_hip | 0.962 | PASS |
| acemagic | 32 | su2_hip | 0.983 | PASS |
| acemagic | 48 | su2_hip | 0.947 | PASS |
| gamer | 8 | su2_cuda | 1.106 | PASS |
| gamer | 16 | su2_cuda | 1.075 | PASS |
| gamer | 32 | su2_cuda | 1.050 | PASS |
| gamer | 48 | su2_cuda | 1.042 | PASS |

## acemagic (backend hip, gfx1151)

### Générateur SU(2)

| L | précision | K | liens/s HB | liens/s OR | pas/s (lot) | pas·chaîne/s | champ (Mo) | réf. pas/s | unifié / réf. |
|---|---|---|---|---|---|---|---|---|---|
| 8 | f64 | 1 | 5.737e+07 | 1.353e+08 | 1.3e+03 | 1.3e+03 | 1 | 1.69e+03 | 0.767 |
| 8 | f64 | 16 | 1.260e+08 | 2.833e+08 | 173 | 2.77e+03 | 8 | — | — |
| 8 | f32 | 1 | 2.869e+08 | 4.380e+08 | 4.84e+03 | 4.84e+03 | 0 | — | — |
| 8 | f32 | 16 | 1.203e+09 | 1.979e+09 | 1.34e+03 | 2.14e+04 | 4 | — | — |
| 16 | f64 | 1 | 1.271e+08 | 2.886e+08 | 176 | 176 | 8 | 182 | 0.962 |
| 16 | f64 | 16 | 1.517e+08 | 3.608e+08 | 13.5 | 216 | 134 | — | — |
| 16 | f32 | 1 | 1.186e+09 | 1.955e+09 | 1.32e+03 | 1.32e+03 | 4 | — | — |
| 16 | f32 | 16 | 1.152e+09 | 1.218e+09 | 57.4 | 919 | 67 | — | — |
| 32 | f64 | 1 | 1.488e+08 | 3.555e+08 | 13.3 | 13.3 | 134 | 13.5 | 0.983 |
| 32 | f64 | 16 | 1.515e+08 | 3.596e+08 | 0.841 | 13.5 | 2147 | — | — |
| 32 | f32 | 1 | 1.068e+09 | 1.113e+09 | 52.6 | 52.6 | 67 | — | — |
| 32 | f32 | 16 | 1.107e+09 | 1.129e+09 | 3.35 | 53.6 | 1074 | — | — |
| 48 | f64 | 1 | 1.490e+08 | 3.622e+08 | 2.65 | 2.65 | 679 | 2.8 | 0.947 |
| 48 | f64 | 16 | 1.523e+08 | 3.573e+08 | 0.166 | 2.65 | 10872 | — | — |
| 48 | f32 | 1 | 1.088e+09 | 1.113e+09 | 10.4 | 10.4 | 340 | — | — |
| 48 | f32 | 16 | 1.080e+09 | 1.107e+09 | 0.648 | 10.4 | 5436 | — | — |

Gain du chemin rapide :

| L | K | F32 / F64 (pas·chaîne/s) |
|---|---|---|
| 8 | 1 | ×3.72 |
| 8 | 16 | ×7.73 |
| 16 | 1 | ×7.52 |
| 16 | 16 | ×4.26 |
| 32 | 1 | ×3.97 |
| 32 | 16 | ×3.99 |
| 48 | 1 | ×3.93 |
| 48 | 16 | ×3.91 |

_fenêtres : 1, durée 527 s, budget 600 s._

### Bain conditionnel

| L | mode | balayage (ms) | balayages/s | cb.py (ms) | gain |
|---|---|---|---|---|---|
| 16 | rapide | 3.03 | 331 | 72 | ×23.8 |
| 16 | ref_rng | 60.1 | 16.6 | 107 | ×1.8 |
| 24 | rapide | 10.9 | 91.4 | 370 | ×33.8 |
| 24 | ref_rng | 245 | 4.09 | 630 | ×2.6 |
| 32 | rapide | 33 | 30.3 | 1.25e+03 | ×37.9 |
| 32 | ref_rng | 675 | 1.48 | 1.97e+03 | ×2.9 |

_fenêtres : 1, durée 189 s, budget 600 s._

### Mesure de fibre

| L | mesure complète (ms) | mesure F seule (ms) | cb.py mesure (ms) | gain |
|---|---|---|---|---|
| 16 | 10.5 | 2.08 | 124 | ×11.8 |
| 24 | 49 | 9.82 | 780 | ×15.9 |
| 32 | 153 | 30 | 2.04e+03 | ×13.3 |

_fenêtres : 1, durée 101 s, budget 600 s._

### Observables

| L | boucles de Wilson (ms) | pas de flot/s |
|---|---|---|
| 16 | 17.9 | 108 |
| 24 | 49 | 29.5 |
| 32 | 139 | 9.78 |

_fenêtres : 1, durée 106 s, budget 600 s._

## gamer (backend cuda, sm_120)

### Générateur SU(2)

| L | précision | K | liens/s HB | liens/s OR | pas/s (lot) | pas·chaîne/s | champ (Mo) | réf. pas/s | unifié / réf. |
|---|---|---|---|---|---|---|---|---|---|
| 8 | f64 | 1 | 6.035e+07 | 1.107e+08 | 1.16e+03 | 1.16e+03 | 1 | 1.05e+03 | 1.106 |
| 8 | f64 | 16 | 2.579e+08 | 4.576e+08 | 302 | 4.84e+03 | 8 | — | — |
| 8 | f32 | 1 | 5.336e+08 | 6.580e+08 | 7.67e+03 | 7.67e+03 | 0 | — | — |
| 8 | f32 | 16 | 1.616e+09 | 4.936e+09 | 2.67e+03 | 4.27e+04 | 4 | — | — |
| 16 | f64 | 1 | 2.566e+08 | 4.568e+08 | 301 | 301 | 8 | 281 | 1.075 |
| 16 | f64 | 16 | 3.015e+08 | 5.306e+08 | 22 | 351 | 134 | — | — |
| 16 | f32 | 1 | 1.545e+09 | 4.242e+09 | 2.4e+03 | 2.4e+03 | 4 | — | — |
| 16 | f32 | 16 | 1.595e+09 | 2.687e+09 | 113 | 1.8e+03 | 67 | — | — |
| 32 | f64 | 1 | 3.010e+08 | 5.302e+08 | 21.9 | 21.9 | 134 | 20.9 | 1.050 |
| 32 | f64 | 16 | 3.054e+08 | 5.347e+08 | 1.39 | 22.2 | 2147 | — | — |
| 32 | f32 | 1 | 1.595e+09 | 2.667e+09 | 112 | 112 | 67 | — | — |
| 32 | f32 | 16 | 1.731e+09 | 2.497e+09 | 6.84 | 109 | 1074 | — | — |
| 48 | f64 | 1 | 3.033e+08 | 5.311e+08 | 4.35 | 4.35 | 679 | 4.17 | 1.042 |
| 48 | f64 | 16 | 3.056e+08 | 5.349e+08 | 0.274 | 4.38 | 10872 | — | — |
| 48 | f32 | 1 | 1.696e+09 | 2.459e+09 | 21.2 | 21.2 | 340 | — | — |
| 48 | f32 | 16 | 1.727e+09 | 2.415e+09 | 1.32 | 21.1 | 5436 | — | — |

Gain du chemin rapide :

| L | K | F32 / F64 (pas·chaîne/s) |
|---|---|---|
| 8 | 1 | ×6.62 |
| 8 | 16 | ×8.83 |
| 16 | 1 | ×7.96 |
| 16 | 16 | ×5.13 |
| 32 | 1 | ×5.11 |
| 32 | 16 | ×4.93 |
| 48 | 1 | ×4.89 |
| 48 | 16 | ×4.81 |

_fenêtres : 1, durée 494 s, budget 600 s._

### Bain conditionnel

| L | mode | balayage (ms) | balayages/s | cb.py (ms) | gain |
|---|---|---|---|---|---|
| 16 | rapide | 1.65 | 604 | 72 | ×43.5 |
| 16 | ref_rng | 76.5 | 13.1 | 107 | ×1.4 |
| 24 | rapide | 6.16 | 162 | 370 | ×60.1 |
| 24 | ref_rng | 462 | 2.16 | 630 | ×1.4 |
| 32 | rapide | 18.6 | 53.8 | 1.25e+03 | ×67.2 |
| 32 | ref_rng | 1.62e+03 | 0.616 | 1.97e+03 | ×1.2 |

_fenêtres : 1, durée 193 s, budget 600 s._

### Mesure de fibre

| L | mesure complète (ms) | mesure F seule (ms) | cb.py mesure (ms) | gain |
|---|---|---|---|---|
| 16 | 6.93 | 1.03 | 124 | ×17.9 |
| 24 | 32.6 | 5 | 780 | ×23.9 |
| 32 | 101 | 15.5 | 2.04e+03 | ×20.2 |

_fenêtres : 1, durée 102 s, budget 600 s._

### Observables

| L | boucles de Wilson (ms) | pas de flot/s |
|---|---|---|
| 16 | 11.1 | 208 |
| 24 | 34.5 | 52.3 |
| 32 | 99 | 17.2 |

_fenêtres : 1, durée 95 s, budget 600 s._

## Lecture

- `liens/s` = 4·V·K ÷ durée d'un sweep ; un pas = 1 bain thermique + 4 OR.
- Références cond/fibre : durées de cb.py (numba, 1 fil, acemagic chargée) du README de noyau2, donc un gain « contre l'existant »,
  pas un banc de même fenêtre.
- Un résultat du gamer est de la RECHERCHE ; ce qui entre dans un verdict est revérifié sur acemagic.


## Contrôle de la cellule rouge : acemagic, L = 8, F64, une chaîne

La gate de non-régression est ROUGE sur acemagic à L = 8 dans la mesure du banc complet (0,767, ci-dessus), et verte à L = 16, 32, 48.
Pour savoir si c'est du bruit de mesure, la même commande a été relancée trois fois, chacune dans sa propre fenêtre de verrou
(`kevinotron bench su2 --L 8 --precision f64 --chaines 1 --budget-s 150 --cible-s 30`) :

| répétition | unifié (pas/s) | su2_hip (pas/s) | rapport | gate |
|---|---|---|---|---|
| banc complet | 1299 | 1694 | 0,767 | FAIL |
| 1 | 1156 | 618 | 1,870 | PASS |
| 2 | 1259 | 707 | 1,780 | PASS |
| 3 | 1202 | 512 | 2,347 | PASS |

Le débit unifié est stable d'une mesure à l'autre (1156 à 1299 pas/s) ; c'est celui de su2_hip qui varie d'un facteur 3
(512 à 1694 pas/s) à L = 8, où un pas dure ~1 ms. La cause n'a pas été établie (état d'horloge du GPU à la prise du verrou, durée courte
de la mesure) : ces quatre mesures ne permettent pas de conclure que l'unifié est plus lent ou plus rapide que su2_hip à L = 8. Le
verdict formel du banc complet reste FAIL ; aucun seuil n'a été modifié et le protocole n'a pas été changé après coup. À L ≥ 16, les
rapports sont de 0,95 à 0,98 (acemagic) et de 1,04 à 1,08 (gamer, L = 8 compris : 1,11).
