# Rendu WP19 — topologie : charge améliorée, flot multi-temps, cooling corrigé (tranche 3)

Implémenteur : Claude (Opus), 26/10/2026. Branche **`wp19`** (arbre `arbres/wp19`, partie de `main` 81521a3), tête = commit de ce rendu ; binaires construits sur **`feb3a22`** (le reste = documents). **Rien n'est fusionné.**
Journal : `journal_tranche3.log`, lignes `WP19`. Gates : `GATES.md` v4.3 (sha256 `457cf117…`) + avenants **v4.3.1** (`89c12948…`), **v4.3.2** (`e0b6ac31…`), **v4.3.3** (`6a16e2e4…`), tous hachés au journal AVANT les calculs qu'ils concernent
(`crates/kvu/validation/topo/AVENANT_v4.3.{1,2,3}.md`, copies `scratch/wp19/AVENANT_*.gele.md`).

## 1. Statut : **PARTIEL — la physique est ROUGE sur TOPO-Q et SANS VERDICT sur TOPO-LIT** (le code, lui, est vert)
| quoi | verdict | une ligne |
|---|---|---|
| **TOPO-Q** (L = 16, β = 2,5, 100 configs, Q amélioré à t/a² = 3) | **FAIL** | **70/100** à |Q − round(Q)| < 0,1 (seuil ≥ 90/100) |
| TOPO-Q-MUT (Q + ½) | PASS | N′ = 4/100, mutant rouge |
| **TOPO-LIT** (AT21, β = 2,427, 16⁴, ⟨Q_I²⟩ = 8,066(64)) | **SKIP (non vert)** | condition (a) : attribution ambiguë, 356/3200 (11,1 %) hors |Q − round(Q)| < 0,25 (il en faut ≤ 2 %) |
| TOPO-LIT-MUT (β × 1,05) | PASS | mutant rouge : X̂ = 2,085, σ = 0,113, z = −46,1 contre 3,261 |
| TOPO-COOL | PASS | 10/10 actions non croissantes |
| TOPO-INST, TOPO-COOL-ID (+ mutant sans dagger rouge), TOPO-GPU | PASS | CPU, HIP, CUDA ; écarts ≤ 1,4e-15 (seuils 1e-10 / 1e-13 / 1e-12) |
| OBS-4-L16 rejouée (INFO) | INFO | **15/20** avec la charge améliorée (5/20 avec le trèfle de v2) |
| WP19-NONREG | voir §6 | |

**Ce que ces deux résultats disent.** La charge améliorée fonctionne (TOPO-INST : erreur 6e-6 à L = 16 contre 4e-4 pour le trèfle ; TOPO-Q passe de 30/100 pour le trèfle à 70/100), mais **à t/a² = 3 le flot est trop court à β = 2,5** :
t₀ moyen de `q` = 5,27 (de 3,44 à 7,94 ; `obs4` 5,48), donc t/a² = 3 ≈ 0,57 t₀ et t/a² = 5 ≈ 0,95 t₀. (Une première lecture du responsable, « t₀ ≈ 3,1 », était fausse : 3,1 est le t₀ de `lit`, β = 2,427 ; correctif au journal.) Les comptes INFO montent avec le temps : t/a² = 1, 2, 3, 5 → 27, 56, 70, 89 sur 100 ; t = 1, 2, 3, 5·t₀ → 85, 95, 98, 98 sur 99. Le seuil de v4.3 n'est donc pas atteint,
et **je n'y ai pas touché** (interdits du brief : pas de changement de temps de flot, pas de relecture d'un FAIL avec plus de données). Pour TOPO-LIT, 20 balayages de cooling ne suffisent pas à atteindre une attribution non ambiguë dans 98 % des cas
(88,9 % à < 0,25 après 20 balayages, en hausse régulière : 71 % à 10, 84 % à 15). La sensibilité au temps de flot et au nombre de balayages est un résultat de WP19, pas un échec d'outil (la référence numpy indépendante et le test croisé HIP/CUDA, §5, vont dans le même sens).
**Aucune de ces constatations ne change un verdict ; toute nouvelle gate (t = 2 t₀, plus de balayages) est un v5 à geler avant de nouvelles données**, sur une nouvelle statistique.

## 2. Lignes brutes (émises par le code ; `validation/topo/topo.json` les recopie telles quelles avec le sha256 du binaire)
Fichiers : `scratch/wp19/analyse_final/sortie_{q,obs4,lit_mut,lit}.txt`, `scratch/wp19/cert/*.lignes`, certificat `crates/kvu/validation/topo/topo.json` (61 lignes ; `non_vert` = `FAIL TOPO-Q`, `SKIP TOPO-LIT`).
**TOPO-Q, TOPO-Q-MUT, TOPO-COOL, INFO (ensemble q)**
```
FAIL TOPO-Q N=70/100 ; seuil 90/100 ; |Q-round(Q)| < 0.100000
PASS TOPO-Q-MUT N′=4/100 ; Q′=Q+0.500000 ; PASS = mutant rouge ; si TOPO-Q passe, N′ ≤ 10 par construction
INFO TOPO-Q-INFO Clopper–Pearson 95 % [0.600185, 0.787594] sur 100 charges finies ; sous hypothèse iid, couverture non garantie (chaînes corrélées)
INFO TOPO-Q-INFO chaîne=1 quantifiées=18/25 disponibles sur 25
INFO TOPO-Q-INFO chaîne=2 quantifiées=19/25 disponibles sur 25
INFO TOPO-Q-INFO chaîne=3 quantifiées=18/25 disponibles sur 25
INFO TOPO-Q-INFO chaîne=4 quantifiées=15/25 disponibles sur 25
INFO TOPO-Q-INFO trèfle non amélioré=30/100 ; accord cooling/flot=65/100 disponibles sur 100
INFO TOPO-Q-INFO histogramme Q ; [-8.000000,-7.750000):0 [-7 … (histogramme complet dans le fichier)
INFO TOPO-Q-INFO histogramme Q-round(Q) ; [-0.500000,-0.4500 … (histogramme complet dans le fichier)
INFO TOPO-Q-T t/a²=1.000000 amélioré=27/100 trèfle=24/100 disponibles sur 100
INFO TOPO-Q-T t/a²=2.000000 amélioré=56/100 trèfle=27/100 disponibles sur 100
INFO TOPO-Q-T t/a²=3.000000 amélioré=70/100 trèfle=30/100 disponibles sur 100
INFO TOPO-Q-T t/a²=5.000000 amélioré=89/100 trèfle=30/100 disponibles sur 100
INFO TOPO-Q-T t=1.000000·t₀ amélioré=85/99 trèfle=32/99 disponibles sur 100 ; absences de t₀ exclues des comptes
INFO TOPO-Q-T t=2.000000·t₀ amélioré=95/99 trèfle=43/99 disponibles sur 100 ; absences de t₀ exclues des comptes
INFO TOPO-Q-T t=3.000000·t₀ amélioré=98/99 trèfle=49/99 disponibles sur 100 ; absences de t₀ exclues des comptes
INFO TOPO-Q-T t=5.000000·t₀ amélioré=98/99 trèfle=60/99 disponibles sur 100 ; absences de t₀ exclues des comptes
PASS TOPO-COOL 10/10 actions non croissantes (tri chaîne,cfg), tolérance 1e-12·|action[k]|
```

**OBS-4-L16 rejouée (ensemble obs4)**
```
INFO OBS-4-L16-REJEU amélioré t/a²=3.000000 : 15/20, 20 configurations ; seuil historique 18/20 ; FAIL historique à titre indicatif
```

**TOPO-LIT (ensemble lit)**
```
SKIP TOPO-LIT (a) attribution ambiguë : 356/3200 hors |Q-round(Q)| < 0.250000 ; X̂=8.807500, σ=0.269557, référence AT21 β=2.427000 L=16 : 8.066000 ± 0.064000 ; 3200 configurations
INFO TOPO-LIT-INFO X̂=8.807500, V=65536, χ_lat=1.343918e-4, ⟨Q_I⟩=0.033750
INFO TOPO-LIT-INFO s=1.139000, X̂′=8.740937, |X̂′-X̂|=0.066563, désaccords=195/3200 ; chaînes figées=[]
INFO TOPO-LIT-INFO ⟨Q_L²⟩=6.702373 (trèfle non amélioré, non arrondi, cooling 20) ; référence AT21 6.635000 ± 0.056000
INFO TOPO-LIT-INFO X̂ flot=9.559375 ; procédure de lissage différente (flot t/a² = 3), sans verdict ; disponibles=3200/3200
INFO TOPO-LIT-INFO τ_int(Q_I²), Sokal c=6.000000 : médiane=0.422208, max=0.606040 (chaînes estimables seulement) ; estimables=32/32 ; non estimable pour chaînes=[]
INFO TOPO-LIT-INFO backend_gen=cuda X̂=9.207500, erreur standard=0.452733, chaînes=16, configurations=1600
INFO TOPO-LIT-INFO backend_gen=hip X̂=8.407500, erreur standard=0.272087, chaînes=16, configurations=1600
INFO TOPO-LIT-PLAQ ⟨½Re Tr U_p⟩=0.636422, σ_jackknife=1.411308e-5 ; référence AT21 0.6364293(15), réseau 20³×16 ≠ 16⁴ ; écart=-6.915281e-6, écart en σ combinés=-0.487247 ; sans verdict, réseau différent
```

**TOPO-LIT-MUT (ensemble lit_mut)**
```
PASS TOPO-LIT-MUT mutant rouge ; X̂=2.085000, σ=0.112934, référence AT21 β=2.427000 L=16 : 8.066000 ± 0.064000 ; 800 configurations ; z=-46.075858 ; quantile Student ν=31 : 3.260942
INFO TOPO-LIT-INFO X̂=2.085000, V=65536, χ_lat=3.181458e-5, ⟨Q_I⟩=0.012500
INFO TOPO-LIT-INFO s=1.093000, X̂′=2.082500, |X̂′-X̂|=0.002500, désaccords=4/800 ; chaînes figées=[]
INFO TOPO-LIT-INFO ⟨Q_L²⟩=1.731743 (trèfle non amélioré, non arrondi, cooling 20) ; référence AT21 6.635000 ± 0.056000
INFO TOPO-LIT-INFO X̂ flot=2.180000 ; procédure de lissage différente (flot t/a² = 3), sans verdict ; disponibles=800/800
INFO TOPO-LIT-INFO τ_int(Q_I²), Sokal c=6.000000 : médiane=0.312182, max=1.267776 (chaînes estimables seulement) ; estimables=32/32 ; non estimable pour chaînes=[]
INFO TOPO-LIT-INFO backend_gen=cuda X̂=2.025000, erreur standard=0.130837, chaînes=16, configurations=400
INFO TOPO-LIT-INFO backend_gen=hip X̂=2.145000, erreur standard=0.187399, chaînes=16, configurations=400
INFO TOPO-LIT-PLAQ ⟨½Re Tr U_p⟩=0.661067, σ_jackknife=2.416917e-5 ; référence AT21 0.6364293(15), réseau 20³×16 ≠ 16⁴ ; écart=2.463740e-2, écart en σ combinés=1017.415593 ; sans verdict, réseau différent
```

**Diagnostic cooling (INFO)**
```
INFO diagnostic (sans verdict, aucune règle changée) : lit, 3200 configurations, fraction de configurations à |Q_amélioré − round| < 0,25 / < 0,1 après k balayages de cooling
INFO cooling k= 0 : amélioré <0,25 : 0.5153 ; <0,10 : 0.2006 ; trèfle <0,25 : 0.5028 ; action moyenne 142964.536
INFO cooling k= 1 : amélioré <0,25 : 0.4953 ; <0,10 : 0.1972 ; trèfle <0,25 : 0.4903 ; action moyenne 19166.702
INFO cooling k= 2 : amélioré <0,25 : 0.5100 ; <0,10 : 0.2009 ; trèfle <0,25 : 0.4913 ; action moyenne 4301.792
INFO cooling k= 3 : amélioré <0,25 : 0.4991 ; <0,10 : 0.2009 ; trèfle <0,25 : 0.5022 ; action moyenne 2225.312
INFO cooling k= 5 : amélioré <0,25 : 0.5075 ; <0,10 : 0.2022 ; trèfle <0,25 : 0.4988 ; action moyenne 1147.988
INFO cooling k= 8 : amélioré <0,25 : 0.6322 ; <0,10 : 0.3150 ; trèfle <0,25 : 0.5016 ; action moyenne 687.757
INFO cooling k=10 : amélioré <0,25 : 0.7106 ; <0,10 : 0.4294 ; trèfle <0,25 : 0.4991 ; action moyenne 554.162
INFO cooling k=15 : amélioré <0,25 : 0.8400 ; <0,10 : 0.6469 ; trèfle <0,25 : 0.5034 ; action moyenne 388.515
INFO cooling k=20 : amélioré <0,25 : 0.8888 ; <0,10 : 0.7472 ; trèfle <0,25 : 0.5334 ; action moyenne 308.914
```

**Contrôle croisé HIP/CUDA (INFO)**
```
INFO contrôle croisé HIP contre CUDA (lit_03, mêmes 100 dumps, mesures indépendantes, sans verdict) : écarts maximaux cooling_q_imp=0.00e+00, cooling_q_tre=0.00e+00, cooling_action=0.00e+00, flot_a2_q_imp=7.11e-15, flot_a2_q_tre=6.22e-15, flot_a2_e_plaq=5.33e-15, plaq0=0.00e+00, t0=1.95e-13, flot_t0_q_imp=3.55e-15, flot_t0_q_tre=2.22e-15 ; Q_I différents : 0/100
```

**X̂ par backend de mesure (INFO)**
```
INFO TOPO-LIT-INFO (avenant v4.3.3) backend_mesure=hip X̂=8.4250 erreur standard=0.8250 chaînes=2 configurations=200
INFO TOPO-LIT-INFO (avenant v4.3.3) backend_mesure=cuda X̂=8.8330 erreur standard=0.2845 chaînes=30 configurations=3000
```

**TOPO-INST (CPU, HIP, CUDA)**
```
PASS TOPO-INST backend=cpu ; L=8 gel_tre=4.440892098500626e-16 gel_imp=0e0 jauge_tre=4.440892098500626e-16 jauge_imp=0e0 reflexion_tre=0e0 reflexion_imp=0e0 ameliore_moins_trefle=7.942608305722576e-3 ; L=12 gel_tre=6.661338147750939e-16 gel_imp=1.3322676295501878e-15 jauge_tre=8.881784197001252e-16 jauge_imp=1.3322676295501878e-15 reflexion_tre=0e0 reflexion_imp=0e0 ameliore_moins_trefle=1.3295351636855024e-3 ; L=16 gel_tre=2.220446049250313e-16 gel_imp=4.4408920985 …
PASS TOPO-INST backend=hip ; L=8 gel_tre=4.440892098500626e-16 gel_imp=0e0 jauge_tre=4.440892098500626e-16 jauge_imp=0e0 reflexion_tre=0e0 reflexion_imp=0e0 ameliore_moins_trefle=7.942608305722576e-3 ; L=12 gel_tre=6.661338147750939e-16 gel_imp=1.3322676295501878e-15 jauge_tre=8.881784197001252e-16 jauge_imp=1.3322676295501878e-15 reflexion_tre=0e0 reflexion_imp=0e0 ameliore_moins_trefle=1.3295351636855024e-3 ; L=16 gel_tre=2.220446049250313e-16 gel_imp=4.4408920985 …
PASS TOPO-INST backend=cuda ; L=8 gel_tre=4.440892098500626e-16 gel_imp=0e0 jauge_tre=4.440892098500626e-16 jauge_imp=0e0 reflexion_tre=0e0 reflexion_imp=0e0 ameliore_moins_trefle=7.942608305722576e-3 ; L=12 gel_tre=6.661338147750939e-16 gel_imp=1.3322676295501878e-15 jauge_tre=8.881784197001252e-16 jauge_imp=1.3322676295501878e-15 reflexion_tre=0e0 reflexion_imp=0e0 ameliore_moins_trefle=1.3295351636855024e-3 ; L=16 gel_tre=2.220446049250313e-16 gel_imp=4.440892098 …
```

**TOPO-COOL-ID (CPU, HIP, CUDA)**
```
PASS TOPO-COOL-ID backend=cpu ecart_max=1.7763568394002505e-15
PASS TOPO-COOL-ID-MUT backend=cpu ecart_max=1.2952536625579972e1 mutant sans dagger
PASS TOPO-COOL-ID backend=hip ecart_max=1.7763568394002505e-15
PASS TOPO-COOL-ID-MUT backend=hip ecart_max=1.2952536625579972e1 mutant sans dagger
PASS TOPO-COOL-ID backend=cuda ecart_max=1.7763568394002505e-15
PASS TOPO-COOL-ID-MUT backend=cuda ecart_max=1.2952536625579972e1 mutant sans dagger
```

**TOPO-GPU (HIP, CUDA)**
```
PASS TOPO-GPU backend=hip trois champs flotés et refroidis ; seuil_delta_q_imp=1e-12
INFO TOPO-GPU-DETAIL backend=hip cfg=/chemin/vers/projet/chantiers/kevinotron_unification/scratch/wp19/ens/q/run_01/cfg_000001.npy mode=flot delta_q_imp=4.440892098500626e-16 delta_q_tre=0e0 delta_E_plaq=0e0
INFO TOPO-GPU-DETAIL backend=hip cfg=/chemin/vers/projet/chantiers/kevinotron_unification/scratch/wp19/ens/q/run_01/cfg_000001.npy mode=cooling delta_q_imp=0e0 delta_q_tre=0e0 delta_E_plaq=0e0
INFO TOPO-GPU-DETAIL backend=hip cfg=/chemin/vers/projet/chantiers/kevinotron_unification/scratch/wp19/ens/q/run_01/cfg_000002.npy mode=flot delta_q_imp=4.440892098500626e-16 delta_q_tre=4.440892098500626e-16 delta_E_plaq=0e0
INFO TOPO-GPU-DETAIL backend=hip cfg=/chemin/vers/projet/chantiers/kevinotron_unification/scratch/wp19/ens/q/run_01/cfg_000002.npy mode=cooling delta_q_imp=0e0 delta_q_tre=0e0 delta_E_plaq=0e0
INFO TOPO-GPU-DETAIL backend=hip cfg=/chemin/vers/projet/chantiers/kevinotron_unification/scratch/wp19/ens/q/run_01/cfg_000003.npy mode=flot delta_q_imp=1.3322676295501878e-15 delta_q_tre=8.881784197001252e-16 delta_E_plaq=0e0
INFO TOPO-GPU-DETAIL backend=hip cfg=/chemin/vers/projet/chantiers/kevinotron_unification/scratch/wp19/ens/q/run_01/cfg_000003.npy mode=cooling delta_q_imp=0e0 delta_q_tre=0e0 delta_E_plaq=0e0
PASS TOPO-GPU backend=cuda trois champs flotés et refroidis ; seuil_delta_q_imp=1e-12
INFO TOPO-GPU-DETAIL backend=cuda cfg=/home/utilisateur/kvu_unif/wp19/ens/q/run_03/cfg_000001.npy mode=flot delta_q_imp=2.220446049250313e-16 delta_q_tre=0e0 delta_E_plaq=0e0
INFO TOPO-GPU-DETAIL backend=cuda cfg=/home/utilisateur/kvu_unif/wp19/ens/q/run_03/cfg_000001.npy mode=cooling delta_q_imp=0e0 delta_q_tre=0e0 delta_E_plaq=0e0
INFO TOPO-GPU-DETAIL backend=cuda cfg=/home/utilisateur/kvu_unif/wp19/ens/q/run_03/cfg_000002.npy mode=flot delta_q_imp=0e0 delta_q_tre=2.220446049250313e-16 delta_E_plaq=0e0
INFO TOPO-GPU-DETAIL backend=cuda cfg=/home/utilisateur/kvu_unif/wp19/ens/q/run_03/cfg_000002.npy mode=cooling delta_q_imp=0e0 delta_q_tre=0e0 delta_E_plaq=0e0
INFO TOPO-GPU-DETAIL backend=cuda cfg=/home/utilisateur/kvu_unif/wp19/ens/q/run_03/cfg_000003.npy mode=flot delta_q_imp=1.4155343563970746e-15 delta_q_tre=1.1657341758564144e-15 delta_E_plaq=0e0
INFO TOPO-GPU-DETAIL backend=cuda cfg=/home/utilisateur/kvu_unif/wp19/ens/q/run_03/cfg_000003.npy mode=cooling delta_q_imp=0e0 delta_q_tre=0e0 delta_E_plaq=0e0
```

## 3. TOPO-Q en détail (INFO)
- Par chaîne (25 chacune) : 18, 19, 18, 15 quantifiées sur 25. Clopper–Pearson 95 % [0,600 ; 0,788] « sous hypothèse iid, couverture non garantie (chaînes corrélées) ».
- Trèfle non amélioré à t/a² = 3 : 30/100. Accord entre Q_I (cooling 20) et round(Q flot t/a² = 3) : 65/100.
- Histogramme de Q − round(Q) (flot t/a² = 3, amélioré) : pic net en [−0,05 ; 0,05[ (60/100) et queue large, 0 configuration au-delà de ±0,5.
- Valeurs par temps : tableau §1 ; les temps k·t₀ sont lus au premier pas de grille ≥ k·t₀ (t₀ par configuration : `q` 99/100 trouvés, moyenne 5,27 ; `obs4` 20/20 : 5,48 ; `lit` 320/320 : 3,15 ; `lit_mut` 63/64 : 7,37 ; deux configurations, `q` chaîne 2 cfg 7 et `lit_mut` chaîne 3 cfg 20, n'ont pas de croisement de t²E = 0,3 jusqu'à t = 50 (plafond de 5 000 pas) : t₀ absent, consigné, elles restent dans tous les comptes à temps fixe).

## 4. TOPO-LIT en détail (INFO, aucun verdict)
- X̂ = ⟨Q_I²⟩ = **8,8075**, σ jackknife (32 chaînes) = **0,2696** ≤ 0,35 (puissance satisfaite : pas d'extension, règle v4.0 ; elle n'est de toute façon déclenchée que par σ), χ_lat = 1,344e-4. **Ce n'est pas un verdict** : la condition (a) échoue en premier.
  Si le verdict avait été évalué, z = (8,8075 − 8,066)/√(0,2696² + 0,064²) = 2,68 contre 3,2609 ; je ne le présente pas comme un PASS.
- Attribution alternative (INFO, utilisée par (c)) : s = 1,139, X̂′ = 8,741, désaccords 195/3200 (6,1 % : la condition (c) échouerait aussi, il en faut ≤ 2 %).
- ⟨Q_L²⟩ (trèfle non amélioré, 20 balayages, non arrondi) = **6,702** contre 6,635(56) d'AT21 : compatible (1,2 σ). X̂ « flot » (Q_I = round(Q à t/a² = 3)) = 9,559 (lissage différent, sans verdict).
- **Plaquette** ⟨½ Re Tr U_p⟩ = 0,636422(14) contre 0,6364293(15) d'AT21 (réseau 20³×16 ≠ 16⁴) : écart −0,49 σ combinés ; l'ensemble est bien celui de la littérature (β et générateur corrects).
- τ_int (Sokal, c = 6) de Q_I² par chaîne : médiane 0,42, max 0,61 (32/32 estimables ; les 100 configurations espacées de 100 pas sont décorrélées) ; aucune chaîne figée.
- Par backend de génération : HIP 8,408 ± 0,272 (16 chaînes), CUDA 9,208 ± 0,453 (16 chaînes) : écart 1,5 σ combinés ; INFO. Par backend de **mesure** (avenant v4.3.3) : HIP 8,425 ± 0,825 (2 chaînes), CUDA 8,833 ± 0,285 (30 chaînes) ; INFO (HIP : 2 chaînes seulement, erreur peu informative ; l'accord des deux backends est établi sur les configurations, §5, pas par ces moyennes).
- Diagnostic cooling (INFO, aucune règle changée, `analyse_final/diag_cooling_INFO.txt`) : fraction à |Q − round| < 0,25 après k balayages : 0,52 (k = 0), 0,51 (5), 0,71 (10), 0,84 (15), **0,89 (20)** ; à < 0,1 : 0,20, 0,20, 0,43, 0,65, 0,75.
  Le trèfle non amélioré reste à ≈ 0,50 tout du long. Action moyenne 143 000 → 309.
- **TOPO-LIT-MUT** : X̂ = 2,085, σ = 0,1129 (puissance OK), conditions (a)(b)(c) satisfaites (désaccords 4/800), z = −46,08 : mutant rouge (la gate voit un β changé de 5 % : ⟨Q²⟩ chute d'un facteur ≈ 4).

## 5. Contrôles indépendants (rien n'est tiré d'un raisonnement)
- **SymPy** (v4_sympy.py, prélude) : valeurs TOPO-INST ; **numpy indépendant** (`scratch/wp19/ref_numpy/topo_ref.py`, matrices 2×2, sans aucun code Rust) : TOPO-INST L = 8 et 12 identiques à SymPy à < 1e-15 ; sur une configuration L = 16 réelle hors gates (graine 26102619999, β = 2,427) :
  plaquette 1e-16, Q trèfle et Q amélioré à ≤ 2,2e-15 et cooling (3 balayages) à ≤ 2,2e-15 contre le Rust GPU (HIP) ; action 6e-11 (annulation 6V − Σ, dite).
- **HIP contre CUDA sur les mêmes dumps** (lit_03, 100 configurations, deux mesures indépendantes) : cooling **identique au bit** (Q amélioré, Q trèfle, action), flot ≤ 7,1e-15, t₀ ≤ 2e-13, Q_I identiques 100/100 (`scratch/wp19/cross/lit03_hip_vs_cuda_INFO.txt`).
- Tests unitaires CPU (`tests/obs_topo_cpu.rs`, 15) : agrafes = somme des 6 plaquettes (matrices 2×2) ; boucles m×m contre produit matriciel ; cooling damier = cooling séquentiel matriciel à 1e-12 ; mutant sans dagger **dégrade la plaquette** (reproduit le bug du 08/10) ;
  TOPO-INST (valeurs v4.3, jauge, réflexion) ; COOL-ID ; flot partiel = complet au bit sur t/a² = 1, 2, 3. Analyse : 33 tests ; GPU : 5 tests (HIP : tous PASS, max|ΔU| flot 9e-16, cooling 0 = identique au bit) ; CLI : 6.

## 6. Non-régression (WP19-NONREG) et fichiers touchés
| gate | acemagic (HIP) | gamer (CUDA) |
|---|---|---|
| `cargo test --release` | **993 passés / 0 échec / 82 ignorés** (938 de `main-t1` + 55 nouveaux : 15 CPU, 33 analyse, 6 CLI, 1 noyaux ; 5 tests GPU ignorés lancés à part), rc 0 | idem avec `--features cuda` : 993 / 0 / 82 |
| `cargo clippy --all-targets -- -D warnings` | rc 0 | 0 avertissement (`--features cuda`) |
| G-H01 | PASS rc 0 (binaire 38ba1d3b, source_hash ce697a3e) | PASS rc 0 (binaire ac622a8a, même source_hash) |
| G-BIT-HIP-PROD / G-BIT-K1 / REGEN-1 | **PASS 3/3** ; K1 L12 6/6, L16 6/6 ; REGEN-1 22/22 + ancres 3/3 (exclusif, MemAvailable 98 Go) | — |
| G-BIT-CUDA | — | **PASS 3/3** (20 s, exclusif) |
| OBS-3 rejouée (`obs_gpu --ignored obs_3`, exclusif) | **PASS** (liens 4,4e-16 ; E 7,4e-15 ; Q 3,9e-16 ; t₀ GPU = CPU 1,139441 ; jauge pure 5,3e-15 ; F32 4,4e-16) | **PASS** (liens 4,4e-16 ; E 8,1e-15 ; Q 4,4e-16 ; jauge pure 5,3e-15 ; F32 5,6e-16) |
| flot inchangé | `obs flot` de `main-t1` (cbb6e3da) contre `wp19` (38ba1d3b), q run_01 cfg 1, HIP, dt 0,01, t = 3 : `flot.csv` **identique à l'octet**, `flot.json` objets égaux | — |
| T0-PROD | PASS (7098f98f, d62ff7d2, cb29307d identiques au début et à la fin) | — |
| `tests/` | diff contre 81521a3 : **4 fichiers ajoutés, aucun modifié** (`obs_topo_{cpu,gates,gpu,cli}.rs`) ; `noyaux_embarques` reste à 12 | |

Fichiers touchés contre 81521a3 (17) : `crates/kvu/Cargo.toml` (feature `float_roundtrip` de serde_json), `README_UNIFIE.md` (+16 lignes), `noyaux/obs_flot.cu` (ajout en fin de fichier), `src/obs/{cli.rs (+3), mod.rs (+2)}`, nouveaux `src/obs/topo.rs`, `src/obs/topo/gpu.rs`, `src/obs/topo_cli.rs`, `src/obs/topo_gates.rs`,
4 fichiers de tests, `validation/topo/{AVENANT_v4.3.1,2,3.md, topo.json}`. Binaires : acemagic `bin/wp19/acemagic/kvu` sha256 `38ba1d3b011ad62a…` (commit feb3a22) ; gamer `~/kvu_unif/bin/wp19/kvu` sha256 `ac622a8a2416a4a2…` (même `source_hash` kvu-source/1 `ce697a3e…`). Sorties brutes : `scratch/wp19/nonreg/`, `scratch/wp19/gates/`.

## 7. Écarts, incidents, décisions (tous au journal)
1. **Avenants** : v4.3.1 (gel avant toute donnée, 5 tours de relecture de GPT-6.1-Sol, constats traités un à un) ; **v4.3.2** : le gel de v4.3.1 s'appuyait sur un débit CUDA de 4,67 pas/s (banc dominé par la compilation à froid) ; les fenêtres réelles ont donné 53-58 pas/s par processus :
   la moitié HIP / moitié CUDA du brief est rétablie (CUDA : `q` 03-04, `lit` 17-32, `lit_mut` 17-32) ; les 4 chaînes CUDA déjà faites sont conservées ; **v4.3.3** : repli de MESURE (non de génération) HIP → gamer CUDA faute de place `gpu_partage` sur l'acemagic. Chaque changement est antérieur à la lecture de toute charge de `lit`/`lit_mut`.
   **La justification « 156 contre 4,67 pas/s » de v4.3.1 est inexacte en valeur** (dit au journal avant tout résultat).
2. **Écart de protocole du brief, déclaré** : flot complet (t/a² = 1, 2, 3, 5 et k·t₀) sur toutes les configurations de `q` et `obs4` ; sur `lit`/`lit_mut` seulement cfg % 10 = 0 ; les autres : flot jusqu'à t/a² = 3 et cooling complet. Aucun verdict n'en dépend (TOPO-LIT lit le cooling).
3. **Incident gpu_partage** (26/10 01:14) : le script a été modifié par un autre lot pendant l'exécution de mes jobs (bash relit au fil de l'eau) : `erreur de syntaxe`, rc 2 ; `lit_23` (génération) et `lit_mut_17/18/21`, `q_03` (mesure) n'ont jamais démarré et ont été relancés tels quels ;
   `lit_29/30/31`, `q_01/02` avaient fini normalement (manifeste `termine`, rc 0, N lignes) et ont été validés sans relance (`valider_incidents.py`).
4. **Collision de mesure** sur `lit_03` (HIP d'origine et repli CUDA) : le fichier retenu est la version CUDA (rsync final), l'autre a été écrasée ; `lit_05` : HIP en échec (fichier déjà présent), version CUDA ; `lit_02` : repli arrêté par PID, version HIP. Vérifié : 69 chaînes, sha256(`topo.jsonl`) = `MESURE_OK`, un seul backend de mesure par fichier, cfg 1…N.
   Mesure : 3450 configurations sur CUDA (dont 1400 de chaînes générées sur HIP : `lit` 03 à 16), 670 sur HIP (`q` 01-02, `obs4`, `lit_mut` 01-16, `lit` 01-02).
5. **Défaut du binaire figé signalé** : un point de reprise de `su2 gen` ne cumule pas les compteurs `plafond`/`essais` d'une fenêtre à l'autre (état aléatoire intact : reprise exacte au bit vérifiée CPU/HIP/CUDA) ; lu après chaque fenêtre (somme du plafond = 0 pour les 69 chaînes). `--reprendre` sans `--backend` repart sur `auto` (autre trajectoire) : `chaine.sh` le passe toujours.
6. **Cargo.toml** (fichier partagé) : `serde_json` avec la feature `float_roundtrip` (relecture exacte des f64 de `topo.jsonl`) ; aucune nouvelle dépendance, `Cargo.lock` inchangé. `obs/cli.rs`, `obs/mod.rs`, `README_UNIFIE.md` : ajouts seuls. `noyaux/obs_flot.cu` : quatre noyaux ajoutés en fin de fichier, le texte historique intact, `obs flot` identique à l'octet à celui de `main-t1` sur une configuration L = 16.
   `git merge-tree` de `wp19` dans `main` 476f4c0 : propre, sans conflit.
7. **Délégations** : GPT-6.1-Sol (code borné, relu et compilé par moi) S1 CPU, S2 GPU, S3 analyse, S4 CLI (specs `scratch/wp19/spec/`, journaux `scratch/wp19/codex_S*.log`) ; 7 relectures Sol des règles (`scratch/wp19/sol/sol_{1..7}.log`). Kimi non utilisé (les coefficients et conventions viennent du prélude : SymPy + Kimi en aveugle, accord total). Pas de Fable. Corrections que j'ai faites moi-même : flot partiel (`ParamsTopo.flot_complet`), tolérance absolue 1e-10 de l'action dans un test (annulation 6V − Σ), égalités exactes pour `s` (avenant), `float_roundtrip`.
8. **Marge des gates** : TOPO-GPU Δ ≤ 1,4e-15 (seuil 1e-12) ; TOPO-INST ≤ 1,4e-15 (1e-10) ; TOPO-COOL-ID 1,8e-15 (1e-13) ; mutants : écart 12,95 (≥ 1e-2) ; TOPO-Q-MUT 4/100.

## 8. Calcul et disque
- Génération : 69 chaînes L = 16 F64 `compat-prod`, ≈ 550 000 pas (277 000 HIP en 35 chaînes, 273 000 CUDA en 34) ; mesure : 4 120 configurations, ≈ 2,14·10⁶ pas de flot RK3 (HIP 0,49·10⁶, CUDA 1,64·10⁶) et 20 balayages de cooling par configuration.
  Débits mesurés : HIP 150-156 pas/s (hors contention), CUDA 53-58 pas/s par processus (4 simultanés, GPU gamer à 100 %). Les places `gpu_partage` de l'acemagic ont été occupées par d'autres lots pendant plus d'une heure : la mesure HIP a été le goulot, d'où le repli v4.3.3.
- Disque utilisé (aucun fichier supprimé) : acemagic `scratch/wp19/ens` 17 Go (35 chaînes HIP) + `topo` 8 Mo ; gamer `~/kvu_unif/wp19/ens` 17 Go (34 chaînes CUDA), **`ens_hip` 12 Go (copies des dumps HIP faites pour le repli : doublons, effaçables sur accord)**, `topo`, `topo_repli`.
  Cibles cargo : `cargo clean` fait : acemagic `cibles/wp19` (696 Mo), `wp19-codex` (1,1 Go), `wp19-codex-s3` (553 Mo) ; gamer `cibles/wp19` (746 Mo) ; binaires `bin/wp19` conservés.
- Machines : MemAvailable 98 Go (acemagic), 26-28 Go (gamer) ; aucune alerte iGPU ; T0-PROD PASS (7098f98f, d62ff7d2, cb29307d identiques au début et à la fin).

## 9. Non vert (liste complète)
1. **TOPO-Q : FAIL 70/100** (seuil 90/100) — résultat de physique, consigné tel quel.
2. **TOPO-LIT : SKIP** (condition (a) : 11,1 % de configurations à attribution ambiguë) — non vert, sans extension possible (σ = 0,27 ≤ 0,35, une extension ne se déclenche que sur l'erreur).
3. Aucun autre FAIL ni SKIP : tous les TOPO-* instrumentaux et WP19-NONREG sont verts. Réserve : la couverture CUDA de `lit`/`lit_mut` repose sur TOPO-GPU (3 configurations par machine), OBS-3 et le contrôle croisé HIP/CUDA de `lit_03` (100 configurations), pas sur une borne universelle.
4. Écarts de protocole déclarés (§7.1-7.2) à arbitrer par le pilote T3.
5. `lit_03` : la version HIP de la mesure a été écrasée par la version CUDA (identiques à 7e-15 près, Q_I 100/100 : le contrôle croisé HIP refait en témoigne) ; les 69 chaînes sont complètes.
