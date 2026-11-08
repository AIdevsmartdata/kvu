# GPU-1 — Portage du générateur SU(2) `su2_hip` vers CUDA (RTX 5060 Ti du PC gamer)

27/09/2026, 17h40-19h10. Plan et critères gelés AVANT les runs : `PLAN.md` (et son addendum, écrit
avant toute analyse). Gates ré-exécutables sur acemagic : `./gates_gpu1.sh` → `logs/gates_run.log`,
**13/13 PASS, RC=0**.

## 0. Réponse courte

- **Le port marche.** `su2_cuda` = `su2_hip` + un en-tête de macros. Le diff avec la source d'origine
  tient en 5 lignes, toutes hors algorithme : include, texte d'erreur, nom du GPU, et les champs
  `engine`/`backend` du manifeste. Mêmes heatbath (Creutz / Kennedy–Pendleton), surrelaxation, damier,
  RNG Philox à compteur et format de sortie (NPY complex128, CSV, manifest). Compilé par
  `nvcc -arch=sm_120 -O3` (CUDA 13.3), RC=0. [calcul exact : `src/port.diff`, sha dans `src/SHA256_PORT.txt`]
- **Mêmes tirages que sur AMD.** Avec la même graine, les moments du test de bain thermique sont
  identiques à l'octet sur 13 des 14 valeurs, la dernière à 1 ulp près. La configuration chaude
  initiale coïncide à 3·10⁻¹⁶ près. La chaîne complète diverge ensuite de façon chaotique, à partir
  d'écarts d'arrondi de la libm (voir P2). [numérique]
- **Physique validée contre acemagic.** Plaquette moyenne sur 6 cellules (L = 6, 8 × β = 2,3 / 2,8 / 3,5) :
  - |z| ≤ 1,83 contre les références AMD de même protocole ;
  - χ² = 4,53 pour 6 ddl (p = 0,61) ; aucun |z| > 2 ;
  - KS par permutation sur les plaquettes locales à β = 2,8 : p = 0,23 (L = 6) et 0,68 (L = 8).
  [numérique, analysé sur acemagic]
- **Débit : ×1,40 par pas à L ≥ 16** face au GPU d'acemagic (bancs du 26/09), parité à L = 12, et
  **×0,62-0,65 aux petits L** (6, 8), où la latence domine. En production L = 6-8, le gamer fait environ
  1050 pas/s. [numérique]
- **Statut** : résultat du gamer = RECHERCHE (règle deux machines). Ce qui peut entrer dans un
  verdict est recalculé sur acemagic : l'intégrité des 360 dumps, les comparaisons statistiques
  contre des ensembles produits sur acemagic, et les sha des 482 fichiers rapatriés.

## 1. Portage (P0)

| élément | valeur |
|---|---|
| source | `src_origine/su2_hip.cpp`, sha ed1e0fce…1e30 = `source_hash` des manifestes AMD [calcul exact] |
| mécanisme | `src/hip2cuda.h` (21 `#define` hip*→cuda*, `hipLaunchKernelGGL(K,G,B,S,ST,...)` → `K<<<G,B,S,ST>>>(...)`) + `src/port.sh` (sed de 5 lignes, vérifie le sha d'entrée) |
| sortie | `src/su2_cuda.cu` sha 510b7fe9… ; binaire `su2_cuda` sha fa4c073c… (`cuda/bin/SHA256SUMS`) |
| compilation | gamer, `~/ym4_calcul/su2_cuda/`, `nice -n 5 taskset -c 12-19 nvcc -arch=sm_120 -O3 -std=c++17`, RC=0 ; 1 avertissement (fonction hôte `polyakov` inutilisée, déjà dans l'original) |
| options numériques | nvcc par défaut : div/sqrt IEEE, FMA contractée ; pas de `--use_fast_math` |
| autotest | KAT Random123 Philox 3/3 PASS, bornes u01 PASS → `SELFTEST OK` [calcul exact] |

Hôte épinglé sur les cœurs E (12-19) pour toutes les exécutions ; le gamer n'a été touché que
dans `~/ym4_calcul/su2_cuda/`. Qwen était déjà arrêté à l'arrivée : `qwen38.service` inactive,
219 Mio occupés sur le GPU, aucun processus de calcul.

## 2. Tests de conformité

### P1 — Tirage exact de x₀ (même test que g1 du chantier source)

7 valeurs de α (Creutz : 0,3 / 1 / 1,999 ; K–P : 2 / 5 / 20 / 100), 2·10⁷ tirages chacune.
Max |z| = 2,11 (m₂ à α = 1,999), seuil gelé à 4 : **PASS** (`cuda/tests/test_heatbath_cuda.json`) [numérique].
Mêmes graines que sur AMD : m₁ et m₂ **identiques à l'octet** pour 6 α sur 7 ; à α = 0,3, m₁
diffère de 1,4·10⁻¹⁷ (1 ulp). Les z sont donc les mêmes que ceux du rapport source [calcul exact].

### P2 — Même graine, même départ chaud, sur les deux GPU (`ref_amd/p2`, `cuda/p2`)

| L | max\|ΔU\| initial | pas 1 | pas 5 | pas 10 | pas ≥ 12 |
|---|---|---|---|---|---|
| 6 | 3,1·10⁻¹⁶ | 2,9·10⁻¹² | 1,5·10⁻⁶ | 6,0·10⁻² | 2 (décorrélé) |
| 8 | 3,5·10⁻¹⁶ | 4,6·10⁻¹² | 3,1·10⁻⁶ | 2,6·10⁻¹ | 2 (décorrélé) |

Gate (initial < 1e-13) : **PASS** [numérique]. Lecture : les entiers Philox sont identiques. Les
écarts viennent des arrondis de la libm (log, cos, sqrt), puis sont amplifiés d'environ ×10-30 par
pas jusqu'à la décorrélation complète (max\|ΔU\| = 2, soit des quaternions opposés) vers le pas 11.
C'est le comportement attendu d'une chaîne MCMC chaotique. **Il n'y a donc pas de reproductibilité à
l'octet ENTRE les machines** ; sur une même machine, elle est exacte (P7). La conformité se juge
statistiquement (P3-P5).

### P7 — Reproductibilité sur le gamer

Relance de L8_b2.8_c1 (22000 pas, 20 dumps) : `measurements.csv`, `therm.csv` et `cfg_020000.npy`
identiques à l'octet (cmp) : **PASS** [calcul exact] (`cuda/logs/p7.log`).

### P6 — Intégrité des 360 dumps CUDA, recalculée SUR ACEMAGIC

- forme (4, L, L, L, L, 2, 2) complex128 : OK ;
- max\|UU† − I\| = 6,7·10⁻¹⁶ ; max\|det U − 1\| = 9,0·10⁻¹⁶ ;
- plaquette numpy = plaquette CUDA du CSV (même indice) à 3,3·10⁻¹⁶ près.

Seuils : 1e-13 / 1e-13 / 1e-12 : **PASS** [numérique] (`analyse/p6.json`). Le code numpy de la plaquette
locale a d'abord été validé sur un dump AMD existant (écart 0,0).

## 3. Validation physique (P3, P4, P5)

Protocole identique sur les deux machines : therm 2000, 20000 mesures, 1 pas = 1 heatbath + 4 OR,
dump tous les 1000 pas, 3 chaînes par cellule (c1 et c2 chaudes, c3 froide). Graines :
27092700000+… (CUDA) et 27092800000+… (AMD neuves). Estimateurs : `stat`/`combine` de
`analyse_plaquette.py` (copie SHA ec5fef48…), τ_int de Sokal (c = 6).

### T1 — Plaquette moyenne, CUDA contre acemagic (z = Δ/σ_comb)

| cellule | CUDA (gamer) | référence AMD | z | τ_int CUDA |
|---|---|---|---|---|
| L6 β2,3 | 0,6028979(53) | 0,6029637(53), validation G2 du 26/09 | −0,87 | 2,0-2,2 |
| L6 β2,8 | 0,6996829(22) | 0,6996983(22), neuve | −0,50 | 0,83-0,85 |
| L6 β2,8 | ″ | 0,6994851(70), production_L6 du 26/09 | +2,69 | ″ |
| L6 β3,5 | 0,7680234(15) | 0,7680254(15), neuve | −0,09 | 0,70-0,75 |
| L8 β2,3 | 0,6023230(28) | 0,6022514(28), validation G2 du 26/09 | +1,83 | 1,8-1,9 |
| L8 β2,8 | 0,6995129(13) | 0,6995224(12), neuve | −0,54 | 0,86-0,91 |
| L8 β3,5 | 0,7679168(9) | 0,7679123(9), neuve | +0,36 | 0,74-0,76 |

Gate P3 (|z| < 3 partout, et χ² sur une référence combinée par cellule avec p > 0,01) :
χ² = 4,53 pour 6 ddl, p = 0,605, 0 cellule avec |z| > 2 : **PASS** [numérique].

**Point à signaler (L6 β2,8, z = +2,69 contre production_L6).** Sous le seuil de 3, mais c'est
une tension **entre les deux références AMD elles-mêmes** : production_L6 contre AMD neuve donne
z = −2,90 [numérique]. production_L6 a peu de mesures (3 × 2000). Ses chaînes étaient déjà dispersées :
z(c1, c2) = +1,98 et z(froid, chaud) = −2,16 [lecture, `plaquette_production_L6.json`, cellule L6_b2.8].
Le τ_int des boucles de Polyakov y valait 37 pas, contre 10-13 ici. À L = 6 et β = 2,8, la boîte est
petite et les secteurs de Polyakov basculent lentement [dérivation]. Avec 2000 mesures, l'erreur de
production_L6 est probablement sous-estimée [conjecture, non testée ici]. La CUDA s'accorde avec la
référence AMD à 20000 mesures (z = −0,50). **Rien à reprocher au port ; la référence production_L6 à
β = 2,8 mérite une barre d'erreur plus prudente si quelqu'un s'en sert.**

### T2 — Contrôles de thermalisation (P4 ; critères C1-C4 du PREREG source)

| cellule | CUDA : z moitiés (c1, c2, c3) | z C2 froid/chaud | z C3 c1/c2 | C4 (1000 ≥ 5τ) | τ_int Polyakov max |
|---|---|---|---|---|---|
| L6 β2,3 | −0,68 / +0,01 / −0,61 | +0,43 | +0,81 | OK | 2,2 |
| L6 β2,8 | +0,13 / +0,06 / +0,36 | +0,42 | +0,55 | OK | 13,0 |
| L6 β3,5 | +1,44 / +0,01 / −1,80 | −0,26 | −0,36 | OK | 19,3 |
| L8 β2,3 | −0,32 / +1,22 / +1,24 | −1,14 | +0,32 | OK | 0,8 |
| L8 β2,8 | +0,15 / +0,52 / −0,15 | −0,04 | +0,94 | OK | 19,1 |
| L8 β3,5 | −0,08 / −0,34 / −1,20 | −1,70 | −0,22 | OK | 31,8 |

Tous |z| < 3 : **PASS** [numérique]. Les références AMD neuves passent aussi C1-C4. Le plus grand z
y est z_C2 = +2,09 à L6 β3,5. Nombre moyen d'essais par lien du heatbath : CUDA 1,03032, contre
1,03031 pour AMD à L8 β2,8 [numérique, logs PERF].

### T3 — KS sur les plaquettes locales ½ReTr U_p à β = 2,8 (P5)

60 configurations par machine (3 chaînes × 20, espacées de 1000 pas), soit 6V plaquettes par
configuration (466 560 à L = 6, 1 474 560 à L = 8), recalculées en numpy sur acemagic. Les
plaquettes d'une même configuration sont corrélées. La p-valeur vient donc d'une permutation des
étiquettes de machine au niveau des configurations (B = 2000, graine 27092026, D sur une grille de 20000 cases).

| L | contre | D exact | p asymptotique naïve (info) | p permutation | médiane de D sous H₀ |
|---|---|---|---|---|---|
| 6 | AMD neuve (gate) | 2,48·10⁻³ | 0,114 | **0,232** | 1,87·10⁻³ |
| 6 | AMD production_L6 (info) | 1,88·10⁻³ | 0,383 | 0,508 | 1,87·10⁻³ |
| 8 | AMD neuve (gate) | 8,69·10⁻⁴ | 0,634 | **0,681** | 1,02·10⁻³ |

Gate (p_perm > 0,01 à L = 6 et 8) : **PASS** [numérique].

## 4. Débit (P8)

`--bench`, β = 2,5, secondes par sweep (temps GPU par événements) ; 1 pas = 1 HB + 4 OR.

| L | CUDA HB | CUDA OR | AMD HB (26/09) | AMD OR (26/09) | gain par pas CUDA/AMD |
|---|---|---|---|---|---|
| 6 | 3,08e-4 | 1,53e-4 | 2,11e-4 | 9,07e-5 | **×0,62** |
| 8 | 3,34e-4 | 1,53e-4 | 2,29e-4 | 9,62e-5 | ×0,65 |
| 12 | 6,10e-4 | 2,98e-4 | 6,54e-4 | 2,73e-4 | ×0,97 |
| 16 | 1,22e-3 | 5,83e-4 | 1,84e-3 | 7,85e-4 | **×1,40** |
| 24 | 5,30e-3 | 2,54e-3 | 8,19e-3 | 3,42e-3 | ×1,41 |
| 32 | 1,62e-2 | 7,96e-3 | 2,41e-2 | 1,09e-2 | ×1,41 |

Sources : `cuda/logs/bench_L*.log` et, côté AMD, `bench/gpu_L*.log` du chantier source,
également cités dans RAPPORT_GPU_GRANDL.md:127-132 [lecture]. En débit de liens, la CUDA atteint
2,6·10⁸ liens/s en HB et 5,3·10⁸ en OR à L = 32.

Lecture et réserves :

- **Le bon point de comparaison AMD est le 26/09.** Le banc AMD refait ce soir (`logs/amd_bench_L*.log`)
  tournait avec acemagic à une charge de 57-69, ce qui le ralentit d'un facteur 2-5 ; il n'est donné
  qu'à titre d'information (gains ×1,9-5,0 contre lui, NON retenus).
- Aux grands L, le gain plafonne vers ×1,4. La RTX 5060 Ti est une carte grand public, au FP64
  bridé à 1/64 du FP32 [conjecture sur la cause, non profilée]. Le code est tout FP64.
- Aux petits L, le temps est dominé par la latence (V/2 = 648 fils par noyau à L = 6) : le GPU AMD
  intégré y est plus réactif. Le CPU du gamer était saturé pendant tous les runs (charge ≈ 17, 20 processus
  python `p_b_mv_bb.py` d'un autre calcul), ce qui peut aussi peser sur les lancements [conjecture].
- **En production** (mesures et CSV compris) : 1028-1100 pas/s à L = 6-8 sur le gamer, contre 470-1000
  pas/s pour les chaînes AMD neuves de ce soir, sur un acemagic chargé [numérique].

## 5. Usage recommandé

- Chaînes L ≥ 16 : le gamer est environ 1,4× plus rapide par pas, **et c'est un second GPU** :
  on peut produire en parallèle d'acemagic sans prendre `kevinotron_gpu.lock`.
- Tout ensemble CUDA destiné à un verdict doit passer : (i) `verif`/P6 sur acemagic (unitarité,
  plaquette recalculée), (ii) au moins une cellule témoin comparée à une chaîne AMD (P3), comme ici.
- Le GPU du gamer ne sert plus Qwen (arrêté à la demande de Kévin). Si Qwen redémarre, les deux
  usages entrent en concurrence sur les 16 Go de VRAM ; L = 32 n'occupe que ~0,15 Go.

## 6. Fichiers

- `PLAN.md` : critères gelés (et addendum d'avant-analyse).
- `src_origine/` : copies SHA (`SHA256_ORIGINE.txt` = original, `SHA256_COPIES.txt`).
- `src/` : `hip2cuda.h`, `port.sh`, `su2_cuda.cu`, `port.diff`, `build_cuda.sh`, `test_heatbath_cuda.py`,
  `gen_commandes.py`, `run_file.sh`, `p7_repro.sh`, `analyse_gpu1.py`.
- `commandes_cuda.txt` (26 commandes), `commandes_amd.txt` (20 commandes) ; RC dans
  `cuda/logs/rc_cuda.log` (26/26 rc=0) et `logs/rc_amd.log` (20/20 rc=0).
- `cuda/` : tout ce qui vient du gamer (ens 18 chaînes, p2, logs, bin, sha distants/locaux : 482
  fichiers identiques).
- `ref_amd/` : chaînes AMD neuves (12) et P2, produites sous `flock /tmp/kevinotron_gpu.lock`, nice 5.
- `analyse/resultats_gpu1.json` : tous les chiffres des tables ; `gates_gpu1.sh` → `logs/gates_run.log`.
- Gamer : `~/ym4_calcul/su2_cuda/` (source, binaire, ens, logs). Rien d'autre n'a été modifié.

## 7. Incidents

- Le premier lancement P2 côté AMD est resté bloqué sur le verrou GPU d'acemagic, tenu de 17h25 à
  18h57 par la suite `cargo test --workspace` de V_KF. Je l'ai arrêté (PID explicites, aucun fichier
  produit), puis relancé comme une file AMD unique sous un seul flock. Celle-ci a obtenu le verrou à
  18h57 et l'a rendu vers 19h03.
- Transfert gamer → acemagic lent par moments (~90 Ko/s : ping LAN de 215 ms, probablement le
  Wi-Fi). Le premier `tar` a été interrompu puis remplacé par `rsync --partial`. L'intégrité finale
  est vérifiée par la liste des sha distants (gate s1).
- `~/ym4_calcul/su2_cuda/` existait déjà, vide (créé à 17h43, avant ma première commande).
