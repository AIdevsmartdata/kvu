# Avenant v4.3.1 à GATES v4.3 — WP19 (responsable WP19, Opus), 2026-10-26

**Nature.** Précisions OPÉRATOIRES de v4.3 : aucune définition de la charge, aucun temps de flot, aucun nombre de balayages, aucune graine, aucun
seuil de v4.3 n'est changé. Il fixe ce que v4.3 laisse à l'implémentation **avant** le lancement de toute génération ou mesure de données du lot
(aucune charge de configuration SU(2) du lot n'a été vue ; seuls des champs jouets L = 4, le champ abélien et une configuration L = 16 de banc
(graine 26102619999, hors gates) ont tourné). Haché au journal (`journal_tranche3.log`, ligne `WP19 GEL avenant`) avant les runs ; relu par GPT-6.1-Sol
(lecture seule), réponses consignées (`scratch/wp19/sol/`). Révision 2 (après `sol_1` : 14 constats, tous pris en compte ci-dessous).

## A. Génération
- Commande, une chaîne par processus : `kvu su2 gen --L 16 --beta B --graine S --chaines 1 --start hot --therm 2000 --nmeas N --skip 100 --nor 4
  --precision f64 --options compat-prod --dump-every 1 --format q64 --backend hip|cuda --outdir <ens>/run_KK` (un dump tous les 100 pas de 1 HB + 4 OR ;
  `su2 gen` prend la première mesure **après** therm + 100 pas : 2 000 + 100·N pas au total). Une chaîne seule = la même chaîne dans un lot (graine propre,
  tirages Philox purs) : vérifié à L = 4 (cmp des trois dumps, `scratch/wp19/essai/g1` contre `g2`). Binaires de génération = ceux de `main` (WP19 ne change pas le
  générateur) : acemagic `bin/main-t1/acemagic/kvu` sha256 `cbb6e3da…`, gamer `~/kvu_unif/bin/main-t1/kvu` sha256 `572f70a3…`.
- Ensembles, graines (v4.3 : graine de la chaîne k = base + k) : `q` β = 2,5, N = 25, graines 26102619101…104 ; `lit` β = 2,427, N = 100, graines 26102619201…232 ;
  `lit_mut` β = 2,54835, N = 25, graines 26102619201…232 (mêmes graines que `lit`, autre β : simulations distinctes mais **non indépendantes** ; le rendu ne présente pas
  `lit` et `lit_mut` comme deux confirmations indépendantes) ; `obs4` (rejeu d'OBS-4-L16, INFO) β = 2,5, N = 20, graine 26102516001. Chaînes numérotées KK = 01…
- **Backend : affectation STATIQUE, fixée ici avant tout run.** Le brief demande « moitié HIP, moitié CUDA » ; v4.3 ne fixe aucun backend. Les mesures de débit du 26/10 sur L = 16, F64, `compat-prod`, 800 pas (`scratch/wp19/essai/b1*`) :
  **HIP 156 pas/s (acemagic), CUDA 4,67 pas/s (gamer, RTX 5060 Ti, FP64 bridé à 1/64)**. 16 chaînes `lit` CUDA = 16 × 12 000 pas = 11,4 h de GPU gamer : la moitié CUDA est abandonnée. **L'avenant fait autorité sur le brief** pour la répartition ; écart déclaré au pilote T3.
  **CUDA** : `q` chaînes 03, 04 ; `lit` chaîne 32 ; `lit_mut` chaîne 32 (≈ 25 500 pas, ≈ 91 min de GPU gamer, les quatre en parallèle sur ses places). **HIP** : tout le reste (dont `obs4`, ≈ 524 000 pas, ≈ 56 min à 156 pas/s hors contention).
  Le verdict de chaque ensemble utilise **toutes** ses chaînes ; X̂ par backend (HIP : 31 chaînes, CUDA : 1) est rapporté en INFO avec cette réserve (un backend à une chaîne ne porte aucune erreur). L'affectation ne change jamais (reprise, extension : même backend, même binaire, mêmes options).
- **Fenêtres ≤ 10 min par processus (brief, v1 §0.7)** : toute chaîne est lancée en fenêtres de **pas** (arrêt PROPRE, pas par signal : un `timeout` tue le processus entre deux écritures et rend le point de reprise incohérent, constaté le 26/10 :
  `measurements.csv : 0 lignes au lieu des 1 attendues`) : première fenêtre `kvu su2 gen … --reprise-every 500 --arret-apres W`, puis fenêtres `kvu su2 gen --reprendre <outdir> --backend <hip|cuda> --arret-apres k·W` (k = 2, 3, …, `--arret-apres` = nombre
  **absolu** de pas faits) jusqu'à `COMPLETE`. **`--backend` est passé à chaque reprise** (sans lui `--reprendre` prend `auto` = le GPU de la machine : trajectoire d'un autre backend, constaté le 26/10 en CPU). W = 12 000 (une seule fenêtre) sur HIP, **W = 1 500** sur CUDA (≈ 5,4 min à 4,67 pas/s ; le banc a montré 253,8 s de mur pour 171,2 s de calcul, soit ×1,5 : ≈ 8 min de mur, sous les 10 min ; si une fenêtre dépasse 10 min de mur, elle est consignée comme dépassement, sans changer le résultat). Fenêtres (la dernière, qui émet `COMPLETE`, peut n'avancer d'aucun pas si l'arrêt demandé tombait exactement sur le dernier pas) : `q` 03/04 et `lit_mut` 32 (4 500 pas) = 4 fenêtres, `lit` 32 (12 000 pas) = 9 ; sur HIP, `lit` = 2 fenêtres (arrêt demandé à 12 000 = dernier pas, puis `COMPLETE`), les autres chaînes (≤ 4 500 pas) 1.
  Règles de la boucle : la présence de `COMPLETE` termine la boucle **avant** tout contrôle de progression (une fenêtre finale sans pas, qui émet `COMPLETE`, est normale) ; pour une fenêtre interrompue, on ne relance qu'après un code retour 0 **et** un `reprise.json` dont `pas_faits` a strictement augmenté ; une fenêtre en erreur ou sans progression arrête la chaîne : ligne `FAIL <chaîne> ECHEC_OUTIL`, comptée `non_vert`, aucune relance aveugle.
  **Contrôle préalable** (avant tout run de données, `scratch/wp19/essai/reprise_test.sh`) : une chaîne L = 8 continue contre la même chaîne interrompue **en thermalisation (pas 45)** puis **en production (pas 100)** et reprise avec `--backend` explicite : les 6 dumps, `therm.csv` et `measurements.csv` identiques **à l'octet** ;
  résultats du 26/10 : CPU PASS, HIP PASS (acemagic), CUDA PASS (gamer) : `identique=1` (arrêts effectifs aux pas 45 et 100 ; l'arrêt a lieu au premier point de reprise ≥ N, d'où W multiple de 500).
  **Compteurs** : le point de reprise ne cumule pas `plafond`/`essais` d'une fenêtre à l'autre (défaut du binaire figé, `compteurs_avant` non resauvegardé, signalé) ; ils ne font pas partie de l'état aléatoire (les dumps sont exacts, contrôle ci-dessus). La ligne `PERF … plafond_atteint=` n'est émise qu'à la fin complète ;
  le plafond HB est donc lu, **après chaque fenêtre (interrompue ou terminée), dans `reprise.json` (`compteurs[0].plafond`) copié sous `reprise_fenetre_<k>.json` dans le dossier de la chaîne avant la fenêtre suivante** (aucun fichier supprimé). Exigence : toutes les lectures présentes, somme des `plafond` = 0, consignée dans le certificat ; une lecture absente
  ou un plafond > 0 rend la chaîne `FAIL ECHEC_OUTIL` (non vert).
- **Lieu de mesure** : les dumps CUDA (gamer) sont **copiés** sur acemagic (liaison directe, `rsync`, sha256 vérifiés des deux côtés) **après `COMPLETE`** de la chaîne source, avec l'inventaire attendu (N dumps) vérifié, dans un dossier de destination propre à la passe (première passe ≠ régénération d'extension : jamais de réutilisation d'une mesure d'une autre passe) ; ils sont mesurés sur HIP avec les autres ; `backend_gen` = `cuda`, `backend_mesure` = `hip` dans le record. La RTX 5060 Ti (FP64 1/64) ne sert à la mesure que pour TOPO-INST, TOPO-COOL-ID, TOPO-GPU (3 configurations) et OBS-3.
- Ordre de lancement (file unique, une chaîne = un job `gpu_partage`, places K = 3 acemagic, K = 4 gamer, dans l'ordre de la file par machine) : `q` 01…04, `obs4`, `lit_mut` 01…32, `lit` 01…32.
  Aucune décision d'ordonnancement ne lit une charge.
- Disque : dumps q64 de 8,4 Mo sous `scratch/wp19/ens/` (acemagic) et `~/kvu_unif/wp19/ens/` (gamer) ; aucun fichier supprimé ; la garde SSD (≥ 100 Go libres) est vérifiée.

## B. Mesure (`kvu obs topo`, une ligne JSON `RecordTopo` par configuration dans `topo.jsonl`)
- Contenu : plaquette brute `plaq0` ; flot RK3 de `obs/flot` (dt = 0,01) sur une copie ; cooling (20 balayages) sur une copie **de la configuration brute** ; charges à chaque point (v4.3).
- **t₀ par configuration** : E_plaq = 24(1 − ⟨P⟩) mesuré à **chaque pas**, t₀ = premier croisement montant de t²E_plaq = 0,3, interpolation linéaire (`flot::echelles`) ; flot poursuivi jusqu'au pas
  max(500, n₅), n_k = ⌈k·t₀/dt − 10⁻⁹⌉ (premier pas de grille ≥ k·t₀) ; Q aux n_k, k ∈ {1, 2, 3, 5}, temps effectif n_k·dt consigné. **Plafond** : 5 000 pas (t = 50, inclus). t₀ non trouvé ou
  n₅ > 5 000 : t₀/`flot_t0` consignés absents (None), jamais d'exclusion de la configuration des verdicts à temps fixe.
- **Économie de flot (optimisation décidée avant les données, sans effet sur aucun verdict)** : les gates **décisives** de `lit` et `lit_mut` (TOPO-LIT, TOPO-LIT-MUT) ne lisent que le cooling ; leurs flots ne servent
  qu'aux lignes INFO. Pour `q` et `obs4` : flot **complet** sur toutes les configurations. Pour `lit` et `lit_mut` : flot **complet** (t/a² ∈ {1, 2, 3, 5} et k·t₀) sur les configurations de numéro `cfg` multiple de 10 (10 par chaîne
  `lit`, 2 par chaîne `lit_mut`) ; sur les autres, flot **partiel** jusqu'à t/a² = 3 (300 pas : valeurs à t/a² = 1, 2, 3 seulement, t₀ non mesuré). **Écart de protocole du brief, déclaré** (le brief écrit « flot … sur chaque configuration ») : économie de calcul demandée par Kévin (« optimisez bien les calculs ») ; la mesure complète sur toutes les 4 000 configurations de
  `lit`/`lit_mut` coûterait ≈ 6,4·10⁶ pas de flot (≈ 5-6 h d'iGPU partagé) contre ≈ 2·10⁶ pas avec l'économie ; **aucun verdict n'en dépend** : le cooling et le flot à t/a² = 3 sont mesurés sur **toutes** les configurations de toutes les lignes. Les INFO « flot » de `lit`/`lit_mut` sont calculées sur les configurations
  où la quantité est mesurée, avec leur effectif écrit. Une **absence planifiée** (flot partiel prévu par la règle cfg % 10) n'est pas une donnée invalide ; une valeur manquante hors de ce plan l'est. Le cooling (20 balayages) est fait sur **toutes** les configurations.
- Backend de mesure : GPU de la machine qui détient les dumps (verrou partagé, ancêtre `flock -s`) ; sortie CPU seulement comme témoin de TOPO-GPU. Arrondi : `f64::round`. Tout est f64 non arrondi jusqu'au verdict.

## C. Précisions de gates
- **Analyse par ensemble, jamais mélangée.** Tout est lu sur l'échantillon FINAL de l'ensemble (extension comprise), configurations dans l'ordre (chaîne, cfg) ; toute configuration absente, dupliquée ou non finie
  rend la ligne concernée SKIP (« donnée invalide »), jamais éliminée en silence (une absence planifiée de la section B n'est pas une donnée invalide).
- **Jackknife « une chaîne retirée »** (chaînes de même longueur) : m_k = moyenne de Q_I² dans la chaîne k ; X̂ = (1/n)Σ m_k ; σ² = Σ(m_k − X̂)²/(n(n−1)) (identique au jackknife leave-one-chain-out de la moyenne) ; n = 32.
  Les chaînes figées sont gardées ; ν = 31 ; aucun jackknife sur les configurations.
- **Seuil de TOPO-LIT et TOPO-LIT-MUT** : le quantile est celui de `v4_student.py` (sha256 `883c3576…`, p bilatéral = 2Φ(−3) = 0,0026997960632602) à ν = 31 : **3,26094191889439** (calcul mpmath 30 chiffres, non arrondi),
  comparaison stricte (`|z| < q` pour PASS ; `|z| > q` pour mutant rouge). La valeur « 3,261 » de v4.3 et « p = 0,0027 » de v4.0 sont des arrondis de cette dérivation ; avec p = 0,0027 exactement le quantile serait 3,26091305 (écart 2,9·10⁻⁵) :
  si un |z| tombait dans cet intervalle, la ligne le dirait en INFO « frontière du quantile », sans changer le verdict.
- **Attribution alternative (c)** : s_k = (800 + k)/1000 calculé en f64 (k = 0…400) ; score(s) = Σ_configs (s·Q − round(s·Q))² accumulé dans l'ordre (chaîne, cfg) par `+=` simple ; minimum strict ; égalités = **égalité exacte de f64** (pas de tolérance),
  départagées par |s − 1| minimal puis s minimal ; une seule optimisation par ligne (`lit`, `lit_mut` séparément), sur **toutes** les configurations de la ligne, jamais par chaîne ni commune aux deux lignes, recalculée sur l'échantillon final.
  Les comptes de (a) (≥ 98 % : 98·n ≤ 100·compte, en entiers) et de (c) (Q_I′ ≠ Q_I : 100·compte ≤ 2·n) sont des comparaisons entières.
- **Ordre de décision et extension (règle v4.0)** : (a), (b), (c), puis puissance, puis critère physique. L'extension est déclenchée **exclusivement** par σ > 0,35 de la première passe (erreur seule, quelle que soit l'issue de (a)(b)(c) et
  quelle que soit la valeur centrale), **indépendamment pour chaque ligne** : `lit` (N = 100 par chaîne) et `lit_mut` (N = 25 par chaîne) ont chacune leur facteur, N′ = ⌈N·(σ/(0,8·0,35))²⌉ arrondi à la dizaine supérieure, les 32 chaînes de la ligne,
  une seule extension ; aucun résultat définitif d'une autre ligne n'est recalculé. Un échec de validité avec puissance satisfaite reste SKIP (ni FAIL physique ni mutant rouge).
- **Extension = régénération contrôlée** : le point de reprise d'une chaîne terminée n'est pas conservé par `su2 gen` avec un N plus grand ; la chaîne est donc **régénérée** de la graine avec N′ (même backend, même binaire, mêmes options, même
  graine, même calendrier de mesures). Équivalence avec une prolongation depuis le point de reprise (tirages Philox purs fonctions de (clé, lien, sweep, tirage) : G-DET) **contrôlée en entier** : les N dumps de la première passe doivent être identiques
  **à l'octet** (`cmp` de chacun) aux N premiers de la régénération ; sinon l'extension est invalide (non vert).
- **TOPO-GPU** : mesure sur chaque machine avec son propre backend : acemagic (HIP) sur les configurations 1-3 de `q` chaîne 01 ; gamer (CUDA) sur les configurations 1-3 de `q` chaîne 03. CPU et GPU exécutent **chacun** le flot (300 pas) et le cooling
  (20 balayages) depuis le **même champ brut** (la comparaison couvre donc flot, cooling et charges). PASS ssi |ΔQ amélioré| ≤ 1e-12 sur les 3 champs flotés et sur les 3 champs refroidis, sur chaque machine. Contrôles complémentaires (|ΔQ trèfle|, |ΔE_plaq|) : lignes INFO,
  **pas des conditions de PASS**. OBS-3 rejouée : `cargo test --release --offline -- --ignored obs_3` sous verrou exclusif sur chaque machine, la ligne `PASS OBS-3` doit figurer dans la sortie (un `SKIP OBS-3` n'est pas vert) ; et `kvu obs flot --cfg <q chaîne 01 cfg 1> --dt 0.01 --tmax 3 --chaque 1 --backend hip`
  du binaire WP19 contre celui de `main-t1` (sha256 `cbb6e3da…`) : `flot.csv` et `flot.json` identiques à l'octet (hors `version` éventuelle, dit tel quel).
- **TOPO-INST / TOPO-COOL-ID** : sur CPU, HIP (acemagic), CUDA (gamer) ; champs construits sur l'hôte puis téléversés ; graines de v4.3. TOPO-COOL-ID appelle la primitive de mise à jour réellement utilisée par le cooling (`maj_cooling` / `k_topo_cool`) ;
  le mutant sans dagger est « rouge » seulement si l'écart ≥ 1e-2 (critère visé) ; la construction des agrafes est couverte par le test unitaire « Σ des 6 plaquettes = Re Tr(U W) » et par la référence numpy indépendante (INFO).
- **TOPO-COOL** : 10 premières configurations de `q` dans l'ordre (chaîne, cfg) ; ΔS ≤ 1e-12·|S| à chacun des 20 balayages ; 10/10.
- **TOPO-Q** : compte sur Q amélioré à t/a² = 3 de **toutes** les 100 configurations. Clopper–Pearson : niveau 95 % bilatéral exact.
- **OBS-4-L16 rejouée** (INFO) : ensemble `obs4` (graine 26102516001, 20 configurations, therm 2 000, espacement 100), charge améliorée à t/a² = 3 (dt = 0,01), compte rapporté à côté de 18/20, sans verdict ; le FAIL historique 5/20 de v2 est conservé.
- **TOPO-LIT-PLAQ** (brief) : moyenne de `plaq0` (jackknife par chaîne) à côté de 0,6364293(15), sans verdict.
- **Sorties** : `validation/topo/topo.json` : lignes de verdict (texte exact émis par le code), sha256 des binaires, commandes, graines, backend de chaque chaîne ; aucun verdict rédigé à la main.

## D. Non-régression (WP19-NONREG)
`cargo test --release` 0 échec ; `cargo clippy --all-targets -- -D warnings` rc 0 (deux machines) ; G-H01 PASS rc 0 ; G-BIT-HIP-PROD 3/3 (acemagic) ; G-BIT-CUDA 3/3 (gamer), **en exclusif** ; T0-PROD ;
`tests/` : ajouts seulement (diff contre 81521a3).
