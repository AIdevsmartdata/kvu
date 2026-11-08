# validation/rg : fixtures et certificat du module RG par blocs (WP6)

- `gen_fixtures.py` : génère UNE fois (venv épinglé `/chemin/vers/projet/outils/ym4_noyaux/venv`, `python3 -I`) les 3 configurations SU(2) 9⁴ (amplitude 0,6, graines numpy 7, 11, 13) et
  leur variable grossière `Ū` (324 liens) calculée par `g_nl.ubar` de S1, exécuté en lecture seule (les instructions de tête de `g_nl.py` jusqu'à `res = {}`, sans l'importer :
  son import exécuterait toutes ses gates et écrirait dans `V/res`). Aucune régénération après le premier hachage.
- `fixtures/` : fichiers `.npy` q64 (4, L, L, L, L, 4), `manifeste.json` (sha256 de chaque fichier, de `g_nl.py`, `geom.py`, `q3_lib.py`, versions). Vérifiés par la gate `RG-FIXTURES`.
- `rapports/` : rapports RG-DOMAINE (`kvu-rg-domaine/1`) des configurations grandL L = 12 (β = 2,3 et 16), 64 classes de θ sur [0, π].
- `rg_bloc.json` : certificat (`kvu-rg-cert/1`) de la batterie complète `kvu rg gates --grandl <ensembles/production> --rapports validation/rg/rapports --certifier`.

Rejeu : depuis `crates/kvu`, `kvu rg gates --grandl /chemin/vers/projet/chantiers/wilson_gpu_grandL_20260926/ensembles/production --rapports validation/rg/rapports`.

## GPU (WP7)
- `gates_rg_gpu.sh` : RG-GPU, RG-DOMAINE-GPU, RG-GPU-MUT, G-DET (module et CLI), ENLIGNE-1 (+ ENLIGNE-1-INCHANGE) sur un backend GPU, sous le verrou ; sorties dans `OUT/sorties/<backend>/`.
- RG-HIP-CUDA : `kvu rg comparer --a <OUT_hip>/sorties/hip --b <OUT_cuda>/sorties/cuda` (les deux dossiers sont produits sur les deux machines puis réunis).
- `kvu rg gates-gpu --backend hip --grandl <ensembles/production>` : la batterie seule.
