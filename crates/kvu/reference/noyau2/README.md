# ym4_noyaux · noyau 2 : chaîne C-b (`cb.py`)

Balayage de bain thermique **conditionnel** (Wilson(β) sachant dec(C) = V fixé) et **mesure de fibre** (F₅₄, D, sommes de blocs).
C'est la version compilée (numba CPU) de `lib_Cb.sweep_cond_dec_fast` et du bloc mesure de `lib_Cb.run_chain`.
Cible n°2 de `../profil/CIBLES.md`.

Références, en lecture seule, copiées à l'octet dans `ref/` (SHA dans `ref/SHA_ref.txt`, vérifiés par les tests) :
`lib_Cb.py` 64967707…, `lib_brancheC.py` 2392599f… (gelé PREREG), `n3_blocage_beta_eff.py` 173b6f9b…, `wilson_hess.py` 86ee757c….
Aucun chantier n'a été modifié.

## Environnement
- acemagic : `/chemin/vers/projet/outils/ym4_noyaux/venv` ; gamer : `~/ym4_calcul/ym4_noyaux_venv`.
  Versions épinglées dans `requirements.lock` : numba 0.67.0, llvmlite 0.49.0, numpy 2.5.3, scipy 1.18.1, Python 3.14.4.
- Fils : `NUMBA_NUM_THREADS` (au plus 8 sur les cœurs E du gamer ; au plus 4, avec nice 10, sur acemagic).
- La première importation compile les noyaux (cache `__pycache__`, `cache=True`) : ~1-2 min sur une machine chargée.

## API
```python
import sys; sys.path.insert(0, '/chemin/vers/projet/outils/ym4_noyaux/noyau2'); import cb

geo  = cb.Geometry(L)                        # masques dec, voisins, paires, plaquettes par lien (une fois par L)
Vfix = cb.V_pairs(C, geo)                    # (4, npair, 2, 2) = lib_brancheC.V_pairs(C, masks_dec(L)) (liste acceptée aussi)
cb.sweep(C, beta, rng, geo, Vfix, mode='ref_rng')   # = sweep_cond_dec_fast, MÊMES tirages numpy (rng = np.random.Generator)
cb.sweep(C, beta, rng, geo, Vfix, mode='rapide')    # tout compilé ; RNG par lien ; équivalence STATISTIQUE seulement
F, D, blk = cb.mesure(C, geo, Wf)            # = (Wf@pv, Gf.T@(Gf/g), block_sums(pv).astype(f32)) de run_chain
F, D = cb.mesure_libre(C, geo, Wf)           # sans conditionnement (tous liens, métrique 1) : témoin Haar G10
A = cb.agrafes(C, geo, mu, sites)            # = lib_Cb.agrafes_at
pv, gp = cb.plaquettes(C, geo)               # ½ReTr U_p et gradients (6L⁴, 4 liens, 3) — ordre loops_plaquettes
Ws = cb.fourier_weights_sparse(L)            # = lib_brancheC.fourier_weights(L, 1) en CSR (obligatoire à L ≥ 24 : dense = 2,7 Go à L=32)
res = cb.chaine(C, beta, rng, Wf, mode=...)  # 100 + 500 balayages, mesure tous les 2, bins de 5 (boucle de run_chain)
```
- `C` : `(4, L, L, L, L, 2, 2)` complex128 **C-contigu** (sinon `ValueError`) ; modifié en place par `sweep`.
- `mode='ref_rng'` : le noyau compilé calcule agrafes, A_eff = A₁ + A₂†V†, k = √det, V = A/k et U = W V†. Le tirage de x₀
  (boucle de rejet) et des directions reste **le code numpy de `heatbath_su2`**, mêmes appels `rng.random`/`rng.normal`, même ordre.
  À β = 0, la référence ne termine pas (α = 0 ⟹ x = NaN, rejet infini) : le noyau lève `ValueError`.
- `mode='rapide'` : un entier 64 bits est tiré de `rng` par balayage (clé) ; chaque lien a son flux splitmix64 (clé, id du lien).
  Même loi x₀ ∝ √(1−x₀²) e^{αx₀} : Kennedy–Pendleton si α ≥ 2, sinon proposition exponentielle + rejet (celle de la référence),
  α < 1e-12 : proposition uniforme (limite exacte à O(α)) ; β = 0 accepté (Haar conditionnel). Direction uniforme sur S² par Marsaglia.
  **Pas d'entier en guise de graine** : une graine répétée rejouerait les mêmes tirages ; on passe un `np.random.Generator` (écart
  voulu à la spéc. « seed » de CIBLES.md).
- `mesure` : Gf = Tᵀ ∂P Wfᵀ n'est jamais formé. Gradients de plaquettes (une passe), puis pour chaque coordonnée de fibre
  (liens libres, métrique 1 ; paires, métrique 2, avec R_ba = −Tr(X_b U₁† X_a U₁)) on accumule la ligne creuse dans D (54×54, un
  accumulateur par tranche). F = `Wf @ pv` est **le même produit numpy** que la référence quand Wf est dense (concordance 1e-13 absolue) ;
  avec `Wf` creux, somme compensée de Neumaier (≈ 1 ulp : plus précise que la réf. dense ; testée contre `math.fsum`).
- Hors périmètre : `kfib_lmin` (λ_min de la Hessienne de fibre) relève de la cible 3 (noyau hess) ; la prolongation PREREG §3.3
  et les estimateurs `cp_var`/`tau_int` restent ceux de `lib_Cb` (bon marché).

## Exemple : une chaîne C-b rapide, estimateurs de la référence
```python
import numpy as np, sys
sys.path.insert(0, '/chemin/vers/projet/outils/ym4_noyaux/noyau2'); sys.path.insert(0, '/chemin/vers/projet/outils/ym4_noyaux/noyau2/ref/code')
import cb, lib_Cb as R, lib_brancheC as B
C = np.ascontiguousarray(np.load('cfg.npy'))                 # config fine ; V = dec(C) est figé par la chaîne
L = C.shape[1]; geo = cb.Geometry(L)
Wf = B.fourier_weights(L, 1) if L <= 16 else cb.fourier_weights_sparse(L)
res = cb.chaine(C, 2.3, np.random.default_rng(2709), Wf, geo=geo, mode='rapide')   # ou mode='ref_rng'
lam, w, _ = R.cp_var(res['F'], res['Db'].mean(0))            # C_P^var de la chaîne
taus = [R.tau_int(res['F'][:, i]) for i in range(res['F'].shape[1])]
assert res['Vdev'] < 1e-12 and res['unit'] < 1e-12            # gates de run_one.py
```
Pour reproduire une chaîne de production à 1e-12 près sur UN balayage (et statistiquement ensuite), utiliser `mode='ref_rng'`
avec la même graine que `run_chain` (`np.random.default_rng(seed)`, le même générateur sert à tous les balayages).

## Validation (code v3, SHA dans `logs/SHA_code_v3.txt` ; `run_gates_acemagic.sh` rejoue tout)
Toutes les gates sont **VERTES sur acemagic** (28/09 ~01h, nice 10, 4 fils, plafond 8 Go ; `logs/RC_gates.txt` : 4 × RC=0).

**Concordance déterministe** (`test_concordance.py`, `logs/concordance_acemagic.json`) : L ∈ {4, 8, 12, 16} ; plat, π-flux
(= `flot_su2.cfg_frustre`), Haar × 3, production L12/L16 b2.3 ; β ∈ {2,3 ; 2,8 ; 16} ; graines {0, 1, 2}.

| gate | n | écart max | tolérance |
|---|---|---|---|
| agrafes (tous ensembles de sites du balayage) | 14 | 2,8e-15 | 1e-14 |
| 1 balayage `ref_rng` contre `sweep_cond_dec_fast` | 100 | 3,5e-14 | 1e-12 |
| GS1 : 1 balayage contre le gelé `sweep_cond_dec` | 93 | 3,5e-14 | 1e-12 |
| Vdev / unitarité après balayage (`ref_rng`) | 100 | 4,6e-15 / 1,3e-14 | 1e-12 |
| Vdev / unitarité après 5 balayages (`rapide`, dont β = 0) | 28 | 4,3e-15 / 6,0e-14 | 1e-12 |
| mesure : F (Wf dense), absolu | 14 | 2,8e-14 | 1e-13 |
| mesure : D, Frobenius relatif (voie BLAS, voie boucle, Wf creux) | 14 | 6,9e-16 / 2,3e-15 / 6,9e-16 | 1e-12 |
| mesure : blk | 14 | identique à l'octet | — |
| F creux (Neumaier) contre la somme exacte `math.fsum` | 14 | 3,3e-16 rel. | 1e-14 |
| F creux contre F réf. : écart ≤ erreur d'arrondi de la réf. elle-même | 14 | toujours | — |
| mesure_libre (G10) : D / F | 14 | 8,1e-16 / 2,8e-14 | 1e-12 / 1e-13 |
| GC1 : valeurs + gradients contre le gelé `plaq_values_grads` | 12 | 1,0e-15 | 1e-14 |
| masques dec, ordre des plaquettes (`plaq_order_check`), n_fib | 6 | identiques | — |
| β = 0 en `ref_rng` : `ValueError` (la référence boucle à l'infini) | 14 | oui | — |

Plat et π-flux : tous les gradients de plaquettes sont nuls (points stationnaires de S_W) ⟹ D_ref = 0 exactement ; la gate D y est absolue.

**Équivalence statistique du mode rapide** (`test_statistique.py`, `logs/statistique_acemagic.json`) contre les **chaînes de
production de la référence** (Cb/runL8, 10 V × β ∈ {2,3 ; 16}, L = 8, copiées avec SHA), même protocole, N égal (250 mesures) :
- F₅₄ par V (z avec τ_int) : β=2,3 |z|max 3,19, 0,74 % au-delà de 3σ (540 z) ; β=16 |z|max 3,45, 0,74 % ; z̄ = −0,015 / −0,078 ;
- C_P^var apparié sur V : t = −0,04 (β=2,3), −0,77 (β=16) ; tr/54 : t = 1,14 / −0,39 ; médianes compatibles à 3σ ;
- **G9 rejouée** : notre recalcul des chaînes de réf. retrouve exactement `test/analyse_L8.json` du chantier (r_P full 1,95867,
  cv 1,05030) ; rapide : full 1,939 (ROUGE, comme la réf. : biais Marchenko-Pastur à N = 250), cv VERT comme la réf., compatibles à 3σ ;
- **G10 rejouée à l'identique** (mêmes tirages Haar, `mesure_libre`) : λ_max et tr/54 à 3e-14 relatif de `gate_G10_GG3b.json`,
  mêmes verdicts (littéral ROUGE, normalisation VERTE, N = 250 et 2000) ;
- loi du bain thermique rapide, intrinsèque : 4·10⁵ tirages par α ∈ {0 ; 0,5 ; 1,9 ; 2 ; 5 ; 30 ; 200} (les deux branches et α → 0),
  moyenne = I₂/I₁ à |z| ≤ 0,69, KS contre la CDF exacte p ≥ 0,25 ;
- **G11 NON rejouée** : elle exige kfib (λ_min de la Hessienne de fibre), c'est-à-dire la cible 3 (noyau hess).

## Vitesse (acemagic, `logs/bench_acemagic_*.json` ; machine CHARGÉE, charge ~80 sur 32 fils : ratios mesurés dans le même processus)
Chaîne standard = 100 + 500 balayages et 250 mesures (run_chain sans kfib). Réf. = lib_Cb, 1 fil.

| L | réf. balayage | cb ref_rng | cb rapide | réf. mesure | cb mesure | chaîne réf. | chaîne ref_rng (gain) | chaîne rapide (gain) | rapide 4 fils (gain) |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 22 ms | 5,2 ms | 2,5 ms | 56 ms | 7,6 ms | 27 s | 5,0 s (×5,5) | 3,4 s (×8,1) | — |
| 12 | 144 ms | 36 ms | 17 ms | 264 ms | 42 ms | 152 s | 32 s (×4,8) | 21 s (×7,3) | 14 s (×11) |
| 16 | 361 ms | 107 ms | 72 ms | 858 ms | 124 ms | 431 s | 95 s (×4,5) | 74 s (×5,8) | 49 s (×8,8) |
| 24 | 2,30 s | 0,63 s | 0,37 s | 9,16 s | 0,78 s | 3671 s | 572 s (×6,4) | 419 s (×8,8) | 220 s (×16,7) |
| 32 | 7,39 s | 1,97 s | 1,25 s | ~29 s (extrapolé en L⁴ ; non mesuré : Wf dense 2,7 Go) | 2,04 s | ~3,2 h (est.) | 1692 s (~×6,9) | 1260 s (~×9,3) | 617 s (~×19) |

Lecture honnête :
- **Un fil par chaîne** (le cas des lots : K chaînes indépendantes en parallèle) : ×4,5-6,4 en `ref_rng`, ×5,8-8,8 en `rapide`.
  La cible ×10-30 de CIBLES.md n'est atteinte qu'avec plusieurs fils par chaîne (×9-17 à 4 fils, sur machine chargée).
- `ref_rng` plafonne à ~×4 sur le balayage : il garde volontairement les tirages numpy de `heatbath_su2` (boucle de rejet vectorisée,
  `rng.normal`) pour rester à 1e-12 de la référence. Le reste (agrafes, A_eff, produits) est ×10 plus rapide que numpy.
- `rapide` : Kennedy–Pendleton pour α ≥ 2 (acceptation → 1 ; la proposition exponentielle de la référence n'accepte que ~1,25/√α,
  soit ~12 % à β = 16) et direction de Marsaglia (sans trigonométrie).
- La mesure n'assemble plus rien : ×6-12, et la mémoire ne dépend plus de 54 × 6L⁴ en dense (Wf creux conseillé dès L = 24).
- Banc gamer : seule une passe sur le code v1 (avant les correctifs) existe (`logs/bench_gamer_1fil.json`, ×3,2-3,7, OBSOLÈTE).
  La passe v2 a été refusée (gamer chargé, charge 26), puis le gamer est devenu injoignable (Tailscale SSH en mode « check » :
  lien de validation à recliquer). Les gains ci-dessus sont des ratios mesurés sur acemagic, dans le même processus.

## CUDA : non fait (décision argumentée)
- Le mode exact (`ref_rng`) ne peut pas aller sur GPU : il consomme la séquence numpy de la référence.
- RTX 5060 Ti : FP64 à ~1/64 du FP32. Un balayage coûte ~500 flops FP64 par lien (agrafes) + la boucle de rejet (log, sqrt) : le gain
  attendu sur une chaîne, face à un cœur CPU numba, est au mieux ×10-20, pour un GPU partagé (verrou commun avec maison3d et la branche C).
- Les lots C-b sont des chaînes INDÉPENDANTES : 8 cœurs E × 1 fil × (×6-9) = le débit utile. Le GPU ne se justifierait que pour de
  longues chaînes à L ≥ 32 ; à reconsidérer si un chantier en demande (le noyau `_hb_rapide` + agrafes se porte tel quel en CUDA).
- Ce soir : gamer injoignable (voir plus haut) et GPU d'acemagic interdit (visionneuse 3D).

## Fichiers
- `cb.py` : le noyau. `run_gates_acemagic.sh` : rejoue toutes les gates + bancs. `test_concordance.py` : concordance déterministe. `test_statistique.py` : mode rapide contre les chaînes de
  production de la référence (L = 8) et gates G9 / G10. `bench.py` : banc de vitesse.
- `ref/` : copies de référence + `SHA_ref.txt`. `data/` : configs de banc (SHA_cfg.txt) et `data/L8/` (V et sorties de production,
  SHA_L8.txt, copiées depuis `sommet_brancheC_20260927/Cb/runL8`).
