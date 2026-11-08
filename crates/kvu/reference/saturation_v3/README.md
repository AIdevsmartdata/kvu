# Référence : lecture tordue de la saturation v3

`torsion.py` est la copie à l'octet de `/chemin/vers/projet/chantiers/brancheC_saturation_v3_20261004/code/torsion.py`
(sha256 dans `SHA256SUMS`, copiée le 04/10/2026 sans toucher au chantier). C'est la définition de Ft et de
Dtw = (D0, Dm, Dp) que `kvu::fibre::torsion` reproduit pour `cond chaine --tordu`.

Fixtures : `validation/cond/gen_torsion.py` (venv ym4_noyaux, noyau2/cb.py) écrit `tw_Ft.npy` et `tw_Dtw.npy`
dans les cas `tests/donnees/fibre/L{4,8}_haar*` ; le test `tests/cond_torsion.rs` les compare au calcul Rust.
