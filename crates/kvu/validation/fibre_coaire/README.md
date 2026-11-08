# validation/fibre_coaire — WP11 : jacobien de la coaire et cas exacts (docs/FIBRE_COAIRE.md)

Ce dossier ne doit pas être confondu avec `validation/fibre/`, qui contient les gates de la mesure de fibre T6 (« paires »).

| fichier | rôle |
|---|---|
| `lancer.sh` | rejoue tout : `bash lancer.sh [sorties]` ; verdicts dans `sorties/verdicts.txt`, empreintes dans `sorties/SHA256SUMS` ; code 0 ssi aucun FAIL ni SKIP |
| `su2_mp.py` | SU(2) en précision arbitraire : formes fermées (log, exp, L_g, L_d, E_g, E_d, jacobien des chemins) ; références génériques (`mp.eig`, `mp.expm`) pour les différences finies |
| `balaban_np.py` | carte de Bałaban L_b = 3 et son jacobien fermé en F64 (énumération des 10 473 chemins réécrite, validée contre les fixtures de WP6) |
| `operateurs.py` | opérateurs linéaires en espace direct (rot, d, Q), loi gaussienne conditionnelle exacte (tranche, Γ, H, M_c), loi pénalisée |
| `t1_dlog.py` … `t8_puissance_mutant.py` | une section du document chacun (table au §10) |
| `kimi/` | seconde dérivation en aveugle (Kimi K3.1 Pro) : prompt, réponse, scripts et sorties de Kimi |
| `sorties/` | sorties du run de référence : `t*.out`, `valeurs_jouet.json`, `valeurs_u1.json`, `valeurs_beta_infini.json` (cas tests gelés), `verdicts.txt`, `info.txt`, `rc.txt`, `versions.txt`, `SHA256SUMS` |

Lectures seules :
- `../rg/fixtures` (sha256 contrôlés contre `manifeste.json`) ;
- le dossier S1 du chantier vision (`KVU_S1`, par défaut `/chemin/vers/projet/chantiers/vision_20261025/calculs/S1_bloc_symetrique_L3`),
  importé en `-B`, sans rien y écrire.
