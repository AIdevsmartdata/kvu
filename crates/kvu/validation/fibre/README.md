# validation/fibre — gates de la mesure de fibre (T6)

| fichier | rôle |
|---|---|
| `gen_fixtures.py` | génère les fixtures depuis `noyau2/cb.py` (venv `ym4_noyaux`) : les 14 cas du plan de `test_concordance.py` + 2 configurations de production L = 8 (bonus). Les configurations de production sont lues EN PLACE (SHA vérifiés). `--petit` écrit le jeu commité (`tests/donnees/fibre`, L ≤ 8, < 5 Mo). |
| `gates_fibre.sh` | FIB-1 (CPU et GPU), FIB-2, FIB-3, puis le banc (L = 16, 24, 32). À lancer sous le verrou GPU de la machine. |

```
# acemagic (HIP) : le jeu de fixtures est généré (numba, ≈ 40 s) s'il n'existe pas ; KVU_FIBRE_FIXTURES le réutilise
flock -w 7200 /tmp/kevinotron_gpu.lock bash crates/kvu/validation/fibre/gates_fibre.sh
# gamer (CUDA), sur le gamer, depuis une copie du worktree (rsync par le câble) et du jeu de fixtures
flock -w 7200 /tmp/gamer_gpu.lock bash crates/kvu/validation/fibre/gates_fibre.sh
```

Variables : `KVU_FIBRE_FIXTURES` (jeu complet existant), `KVU_FIBRE_GARDER=1` (garde les fixtures générées),
`KVU_FIBRE_BANC=0` (saute le banc), `KVU_FIBRE_BANC_L=16,24,32`, `KVU_FIBRE_PROFIL=1` (durée de chaque étape du banc GPU),
`KVU_PYTHON`, `KVU_NOYAU2`. Le script s'arrête sous 110 Go libres (acemagic) ou 50 Go (gamer).

Seuils gelés (ARCHITECTURE_UNIFIEE.md §6.3) : F contre `cb.mesure` (Wf dense) ≤ 1e-13 absolu ; F de Neumaier contre
`math.fsum` ≤ 1e-14 relatif ; D ≤ 1e-12 en Frobenius relatif ; blk identique aux bits float32 ; `mesure_libre` D ≤ 1e-12 et
F ≤ 1e-13 ; plaquettes et gradients ≤ 1e-14 ; plat et π-flux : D = 0 exactement. FIB-2 : D GPU identique à l'octet sur 2
exécutions et pour 3 géométries de lancement. FIB-3 (en plus de la spec) : GPU = CPU à l'octet (pv, gradients, F, D, blk,
mesure libre), sur les fixtures, sur L = 6 et 10 et, dans le banc, à L = 16, 24 et 32.

Lancer à la main (jeu commité, CPU) : `cargo test -p kvu --test fibre_cpu`.
Mesure d'une configuration : `kvu fibre mesurer --cfg X.npy [--libre] [--backend auto|hip|cuda|cpu] --out DIR`
(`F.npy`, `D.npy`, `blk.npy`).
