# validation/cond — gates du bain thermique conditionnel (T5)

| fichier | rôle |
|---|---|
| `gen_fixtures.py` | fixtures de `noyau2/cb.py` (géométrie, numpy, `_hb_numpy`, balayages `ref_rng`) ; niveau `petit` = jeu commité (`tests/donnees/cond`, L = 4, < 5 Mo), niveau `complet` = ≥ 100 cas à L = 4 et 8 (TEMPORAIRE, jamais commité) |
| `gen_tables_numpy.py` | contrôle que `src/cond/numpy_tables.rs` est identique à `ziggurat_constants.h` de numpy 2.5.3 |
| `gates_cond.sh` | EQC-1, EQC-1b, EQC-3, EQC-4 (Rust, GPU), puis EQC-4 CLI et EQC-2 si la fibre de T6 répond ; à lancer SOUS le verrou GPU |
| `eqc2.py` | EQC-2 : mode `rapide` contre les chaînes de production L8 (protocole et seuils de `test_statistique.py`), loi du tirage, mutant β × 1,01 ; `--decalage-graine` ≠ 7919 = réplicat d'information (non gaté) |
| `banc_cond.sh` | banc de la chaîne standard (100 + 500 balayages, 250 mesures) à L = 16, 24, 32 contre le README de noyau2 |

Commandes (acemagic ; gamer : verrou `/tmp/gamer_gpu.lock`, `KVU_FEATURES=cuda`, fixtures copiées) :

```
cargo test -p kvu --release --test cond_geometrie --test cond_numpy --test cond_cpu --test cond_chaine     # CPU, aucun GPU
flock -w 7200 /tmp/kevinotron_gpu.lock bash crates/kvu/validation/cond/gates_cond.sh                       # GPU, < 10 min
flock -w 7200 /tmp/kevinotron_gpu.lock bash crates/kvu/validation/cond/banc_cond.sh 16 24 32               # banc
```

`KVU_COND_FIXTURES=<dossier>` : jeu complet (généré par `gates_cond.sh` s'il manque et si le venv numba est là).
`KEVINOTRON_SSD_MIN_GO=0` : seuil de la garde SSD pour les essais écrits dans `/tmp` (tmpfs).
Tant que le binaire unifié n'est pas câblé (T9), les scripts appellent la ligne de commande par le « pont » `tests/cond_cli.rs`.
