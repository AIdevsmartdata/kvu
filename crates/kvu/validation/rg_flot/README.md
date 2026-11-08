# validation/rg_flot : étalonnages, flot.json et certificat des estimateurs de couplage effectif (WP8)

Protocoles gelés au journal de la tranche 1 (ligne « WP8 GEL des protocoles », 25/10/2026 21:00, + erratum de graines) AVANT tout run.

- `etalonnages/etal_L4.json` : courbe d'étalonnage de Wilson au volume grossier L = 4 : 14 β ∈ {1,8 ; 2,0 ; 2,2 ; 2,4 ; 2,6 ; 2,8 ; 3,0 ; 3,5 ; 4 ; 5 ; 6 ; 8 ; 12 ; 16}, 3 chaînes par point (graines 26102608000 + 100·i + c, c = 1, 2, 3),
  départ chaud, 2 000 pas de thermalisation, 20 000 mesures (un pas = 1 bain thermique + 4 OR), mesures en ligne de P (1×1) et R (1×2). `kvu rg etalonner --L 4 --sortie …`
- `etalonnages/indep_L4.json` : chaîne indépendante de ré-estimation (c = 9) de RG-PLAQ. `kvu rg etalonner --L 4 --independante --sortie …`
- `etalonnages/indep_mut_L4.json` : mutant de RG-PLAQ (chaîne indépendante simulée à 1,05 β). `kvu rg etalonner --L 4 --independante --facteur-beta 1.05 --sortie …`
- `etalonnages/etal_L6.json` : supplément HORS GATE (L = 6, graines 26102610000 + 100·i + c) pour le rapport RG-SCHEMA du témoin de Swendsen (L = 12 → 6).
- `flot_L12.json` : flot.json de validation, grandL L = 12, Bałaban un pas (β = 2,3 ; 2,5 ; 2,8 ; 4 ; 8 ; 16, 3 chaînes × 20 configurations), estimateurs sd, plaq, mcrg2 ; sorties rapportées, aucune lecture physique.
- `rg_flot.json` : certificat (`kvu-rg-flot-cert/1`) de la batterie complète.

Rejeu depuis `crates/kvu` (acemagic, CPU) :

```text
kvu rg etalonner --L 4 --sortie validation/rg_flot/etalonnages/etal_L4.json
kvu rg flot-gates --grandl <ensembles/production> --etalonnages validation/rg_flot/etalonnages --certifier
kvu rg flot --ensemble <ensembles/production> --filtre 'L12_b*_c*' --courbe validation/rg_flot/etalonnages/etal_L4.json --sortie validation/rg_flot/flot_L12
```

Références Python (lecture seule, `tests/donnees/rg_flot/`) : `n3_ref_ape_…json` et `n3_ape_niv1_L6.npy` (sortie de `n3_blocage_beta_eff.py`), `tau_ref.json` (`analyse_plaquette.py`),
`bessel_ratios.json` (mpmath, 60 chiffres) ; les scripts générateurs `gen_tau_ref.py`, `gen_bessel.py` sont à côté.
