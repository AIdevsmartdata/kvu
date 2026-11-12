# Défauts connus de kvu 0.1.0 (état dans la 0.1.1)

## 1. Charge topologique de `obs flot` : non quantifiée à β = 2,5 (gate OBS-4-L16 déclassée)

**Statut** : observable **expérimental, défaut connu**. La gate `OBS-4-L16` (test GPU `obs_gpu::obs_4_l16_quantification`)
échoue sur les deux backends (passe GPU du 8 novembre 2026) : **5/20** configurations sur CUDA (RTX 5060 Ti ; rejeu identique au
bit) comme sur HIP (Radeon 8060S ; la chaîne générée diffère d'un backend à l'autre) ont `|Q − round(Q)| < 0,1` après un flot
de Wilson jusqu'à t/a² = 3 (L = 16, β = 2,5, 20 configurations d'une chaîne, graine 26102516001), pour un seuil gelé de
**18/20**. Elle est **déclassée** (échec documenté, non bloquant) pour la publication 0.1.0 ; ni le test ni son seuil n'ont été
modifiés, et elle reste rouge.

**Ce que mesure la gate.** `kvu obs flot` donne `Q_top` par la définition « trèfle » (clover) non améliorée,
Q = −(1/(32π²)) Σ_x ε_{μνρσ} tr(G_μν G_ρσ) (`crates/kvu/src/obs/ref_cpu.rs`), à chaque mesure du flot.

**Pourquoi ce n'est pas une erreur de normalisation, de signe ou de flot** :

- même calcul sur GPU et sur la référence CPU : écart 4e-16 (gate OBS-3) ;
- normalisation : sur un instanton discrétisé, la charge trèfle vaut 1 à 4e-4 près à L = 16 (gate TOPO-INST,
  `crates/kvu/validation/topo/RENDU_WP19.md`) ; après 20 balayages de cooling, ⟨Q_L²⟩ (trèfle non arrondi) = 6,702 contre
  6,635 ± 0,056 pour Athenodorou et Teper à β = 2,427 sur 16⁴ (`crates/kvu/validation/topo/topo.json`). Une erreur de facteur k
  donnerait k² ; le rapport est 1,01 ;
- un signe faux n'empêcherait pas la quantification ; le flot n'est pas en cause : `obs topo` emploie le même pas RK3 de
  Lüscher (`pas_rk3` de `ref_cpu.rs`) et sa charge améliorée, calculée sur les configurations flotées, se quantifie (ci-dessous) ;
- prolongé jusqu'à t/a² = 12, le flot ne rend pas les charges entières : la plupart se stabilisent à environ 0,86 à 0,96 fois
  un entier, **en dessous** de cet entier en valeur absolue (−0,88 ; 0,93 ; −1,79 ; 1,73 ; −2,69 ; −2,82 ; 3,79…), et
  quelques-unes dérivent encore (1,11 → 1,29 ; −1,64 → −0,69) ; diagnostic hors gate `obs_4_l16_diagnostic_flot_long`, lignes
  `INFO OBS-4-L16-DIAG` (journaux de la passe GPU du 8 novembre 2026).

**Explication.** C'est l'effet de discrétisation connu de la charge non améliorée. Athenodorou et Teper (arXiv:2106.00364v3,
§ sur la topologie, commentaire de leur figure 20) écrivent que le pic des charges refroidies n'est pas sur un entier parce que
« for an instanton of size ρ on a lattice, we only obtain Q_L → 1 in the limit ρ/a → ∞, with deviations from unity that are
powers of a/ρ », et que « the discretisation corrections are negative ». Ils attribuent une charge entière Q_I à chaque
configuration plutôt que d'exiger Q_L entier. Le critère « 0,1 d'un entier pour 90 % des configurations, à t/a² = 3, avec la
charge trèfle » est donc plus strict que ce que cette définition donne à β = 2,5 ; aucun seuil n'a été changé pour autant.

**Ce qui marche.** La charge **améliorée** de `kvu obs topo` (module WP19) se quantifie : 70/100 à t/a² = 3, 89/100 à 5,
95/99 à 2 t₀ et 98/99 à 3 t₀, contre 30/100 à 49/99 pour le trèfle sur les mêmes configurations (`RENDU_WP19.md`, lignes
`TOPO-Q-T`). Pour une charge entière, utiliser `kvu obs topo` et un temps de flot exprimé en t₀ ; le `Q_top` de `obs flot` est
une sortie de contrôle.

**Suite prévue (0.2)** : remplacer, dans `obs flot`, la charge trèfle par la charge améliorée (ou ajouter un arrondi
documenté), et reformuler la gate en conséquence, avec un nouveau critère gelé avant tout calcul.

## 2. Gates et tests GPU qui ne tournent qu'avec leur script de validation

- Les tests F32 de `tests/su2_f32_gpu.rs` ne font que `SKIP` hors de `validation/f32/gates_f32.sh` (variable
  `KVU_F32_SOUS_VERROU=1`) ; les gates de loi LOI-0 à LOI-4 sont une campagne pilotée par ce script.
- `G-BIT-K1` demande les références `su2_gen` (`KVU_REF_K1`) ; `MACH-1` et `DETECTION-REELLE` demandent un `kvu.toml` qui
  nomme la machine réelle (la fixture `crates/kvu/kvu.toml` a été neutralisée pour la publication).

## 3. Défauts de la 0.1.0 relevés le 11 novembre 2026 (dix-huit points) et leur statut dans la 0.1.1

Relevé par relecture et reproductions sur l'arbre public 3163bd9 (liste complète avec preuves : fichier interne `DEFAUTS_PUBLICS_v0.1.0.md`).
Aucun résultat numérique faux n'a été trouvé dans le domaine d'usage : la chaîne de Markov, `measurements.csv` et les dumps ne sont
pas concernés par PUB-01 à PUB-16. Deux limites AUX MARGES touchent en revanche la chaîne de Markov elle-même, hors de tout usage
connu : PUB-18 (plus de 2³² balayages dans un run) et PUB-19 (β extrêmement petit) ; elles sont déclarées ici, non corrigées.
« Corrigé » = corrigé dans la 0.1.1 ; « documenté » = connu, non corrigé, décrit ici.

| id | défaut | gravité | statut 0.1.1 |
|---|---|---|---|
| PUB-01 | `su2 gen --reprendre` sans repasser `--mesures` clôturait (COMPLETE) avec des fichiers de mesure en ligne tronqués ; une spécification différente était acceptée et mélangée | échec silencieux | corrigé : la spécification est enregistrée (`mesures_en_ligne.json`) et rechargée ; une option différente est refusée (code 2) |
| PUB-02 | `su2 gen --bord --mesures …` rendait COMPLETE sans aucune mesure | échec silencieux | corrigé : refus, code 2 |
| PUB-03 | L'admission mémoire lit `MemAvailable` de `/proc/meminfo` et ignore un plafond cgroup (conteneur, `MemoryMax`) | échec silencieux | documenté : en conteneur, l'admission voit la RAM de l'hôte ; ne pas lancer de run dont le besoin dépasse la limite du conteneur |
| PUB-04 | `scripts/build_auto.sh` : la garde de la cible de production portait un chemin fictif (épuration) | garde inopérante | corrigé : `*/kevinotron/target*`, testé avec mutant |
| PUB-05 | Les CSV de mesure ne sont pas fsyncés avant le point de reprise | coupure système | documenté (à confirmer) : après coupure du courant, la reprise est refusée bruyamment (« N lignes au lieu des M attendues »), sans donnée fausse |
| PUB-06 | `validation/historique/source_hash_kvu.py` exige Python ≥ 3.11 (`hashlib.file_digest`) ; 6 tests rouges en 3.10 | portabilité | documenté : utiliser Python ≥ 3.11 pour la suite complète ; aucun effet sur les sorties de `kvu` |
| PUB-07 | `su2_cpu::golden_des_csv_et_des_dumps` (1e-9) et `obs_cpu::regression_valeurs_gelees` (1e-10) dépendent de la libm : rouges sur un autre glibc | portabilité | documenté : valeurs figées sur la machine de l'auteur ; la reproductibilité au bit vaut pour un même binaire sur une même machine |
| PUB-08 | Sous `flock -F <verrou_de_la_politique> kvu …`, le verrou GPU attend son propre détenteur jusqu'au délai (code 6) | portabilité | documenté : utiliser `flock` sans `-F` |
| PUB-09 | Les tests `detection_reelle_sur_cette_machine` et `MACH-1` comparaient au nom d'hôte (fictif depuis l'épuration) et se sautaient | tests | corrigé : `KVU_TEST_MACHINE_ATTENDUE=acemagic|gamer` ; sans elle, SKIP annoncé |
| PUB-10 | `besoin_memoire` de `su2 gen` compte un forfait pour les mesures en ligne, pas leur espace de travail réel | tests | documenté (à confirmer) : prévoir de la marge sur GPU avec `flot` en ligne |
| PUB-11 | Chaînes HIP et CUDA de même graine différentes (reproductibilité au bit = même backend) | documentation | documenté ici |
| PUB-12 | `cargo fmt --check` : 3 048 blocs non conformes | cosmétique | documenté |
| PUB-13 | Après un arrêt BRUTAL (kill, panne) entre deux points de reprise, `--reprendre --mesures` rajoutait les mesures postérieures au point : lignes en double, run COMPLETE | échec silencieux | corrigé : les fichiers des mesures en ligne sont ramenés au point de reprise, préfixe validé avant toute troncature |
| PUB-14 | Après un arrêt BRUTAL du processus (kill -9, plantage) pendant la thermalisation, `su2 gen --reprendre` refusait le run (code 4, « measurements.csv : 0 lignes au lieu des 1 attendues au point de reprise ») : l'en-tête de `measurements.csv` (avec `--bord` : de `blocs.csv`, et de `therm_blocs.csv` juste après sa création) restait dans un tampon | refus bruyant | corrigé : les en-têtes sont écrits et synchronisés dès leur création (tests à vrai `kill -9`, chemin standard et `--bord`) ; la reprise repart du dernier point de reprise valide ; avant le premier point de reprise, il n'y a rien à reprendre ; une coupure de courant relève de PUB-05. Parade pour la 0.1.0 : si `measurements.csv` est VIDE (0 octet), y écrire l'en-tête, `echo "measurement_index,total_steps,phase,plaquette,P0,P1,P2,P3" > DIR/measurements.csv` (pour un lot : dans chaque `DIR/cNNN/` concerné) ; avec `--bord`, pour CHACUN des deux fichiers `blocs.csv` et `therm_blocs.csv` qui existe et est VIDE (les deux peuvent l'être ; même en-tête) : `for f in blocs.csv therm_blocs.csv; do [ -e DIR/$f ] && [ ! -s DIR/$f ] && python3 -c 'import sys;L=int(sys.argv[1]);print(",".join(["bloc"]+["S%d"%t for t in range(L)]+["T%d"%t for t in range(L)]))' L > DIR/$f; done` (L = côté du réseau ; pour un lot : dans chaque `DIR/cNNN/` concerné) ; puis reprendre ; la condition de la boucle n'écrit que dans un fichier vide : ne jamais écraser un fichier non vide |
| PUB-15 | `kvu regen` ne rejoue pas un run `su2 gen` : son `manifest.json` n'est pas reconnu (code 4, « manifeste non reconnu ») ; le `regen_bloc3.json` écrit par `--mesures bloc:3` est reconnu mais son rejeu échoue (FAIL REGEN, code 8 : la commande enregistrée garde `--outdir`, que `regen` ajoute une seconde fois). Le README de la 0.1.0 annonçait des configurations « régénérées au bit près » | documentation | documenté : `kvu regen` rejoue les manifestes `kvu-regen/1` de `su2 boite` et de `fibre echantillonner` ; pour un run `su2 gen`, relancer la commande enregistrée dans son manifeste et comparer les sha256 (README, « Limites connues » ; GUIDE, « Régénération des ensembles ») |
| PUB-16 | Un run REPRIS (`--reprendre`) puis terminé portait dans `COMPLETE` une phrase fixe au lieu de `rc=0` et de l'empreinte du manifeste : `kvu store verifier` le déclarait rouge (code 8, « COMPLETE annonce , manifest.json vaut … ») ; les données ne sont pas en cause | vérification | corrigé : `COMPLETE` d'un run repris a le format d'un run neuf (`rc=0`, `manifest_sha256=`), testé par `kvu store verifier`. Parade pour un run repris par la 0.1.0 (lecture seule) : `python3 -c 'import json,sys;[print(f["sha256"]+"  "+sys.argv[2]+"/"+f["nom"]) for f in json.load(open(sys.argv[1]+"/manifest.json"))["fichiers"]]' DIR DIR \| sha256sum -c` vérifie les fichiers consignés ; pour que `store verifier` passe, réécrire `COMPLETE` d'un run dont le manifeste dit `"statut": "termine"` : `printf 'rc=0\nmanifest_sha256=%s\n' "$(sha256sum DIR/manifest.json \| cut -d' ' -f1)" > DIR/COMPLETE` |
| PUB-18 | Au-delà de 2³² (4 294 967 296) balayages de bain thermique dans un même run (total standard : `--therm` + `--nmeas` × `--skip` ; avec `--bord` : `--pre` + `--therm` + `--nmeas` ; reprises comprises ; le compte vaut pour chaque chaîne séparément, chaînes auxiliaires comprises), le compteur de balayages du générateur aléatoire, sur 32 bits, repasse à zéro : les nombres aléatoires du début du run sont réemployés, sans refus ni avertissement. Tous les moteurs du bain thermique SU(2) sont concernés (CPU, F64 de référence, F32 rapide). Aucun run de cette longueur n'est connu : à 200 balayages par seconde (ordre de grandeur mesuré à L = 16 sur GPU) il faut 248 jours ; un très petit réseau y arrive plus vite | validité hors domaine | documenté : ne pas dépasser 2³² balayages par chaîne (au-delà : plusieurs chaînes de graines différentes) ; refus explicite prévu en 0.2 |
| PUB-19 | À β extrêmement petit, le tirage F64 du bain thermique (algorithme de Creutz, CPU et noyau GPU de référence) dégénère. Avec k ≤ 6 la norme de la somme des agrafes d'un lien : dès que β·k passe sous environ 2,8·10⁻¹⁷ (β = 0 compris), `exp(−2βk)` vaut exactement 1 en double précision et aucun tirage n'est accepté : chaque mise à jour qui passe par Creutz épuise son plafond de 10⁶ essais et laisse le lien inchangé (les liens gelés ne sont pas mis à jour ; si k ≤ 10⁻³⁰⁰ le lien est tiré selon la mesure de Haar, sans Creutz). Le run est alors très lent et n'échoue qu'à la FIN (code 8, pas de `COMPLETE`) : `su2 gen`, avec ou sans `--bord`, dit « plafond d'essais atteint N fois » ; `su2 boite` ajoute, à tort dans ce cas, « champ probablement corrompu (faute GPU) ». Avec une torsion, β = 0 est refusé d'avance (code 2). Juste au-dessus du seuil, le tirage ne prend plus qu'un petit nombre de valeurs distinctes (ordre de grandeur : 2βk / 1,1·10⁻¹⁶). Le chemin F32 rapide n'est pas concerné (branche dédiée aux petits β·k) | validité hors domaine | documenté : rester à β ≥ 10⁻³ ; domaine de β borné et message exact prévus en 0.2 |

PUB-17 (rejeu de `regen_bloc3.json`) est décrit avec PUB-15. PUB-18 et PUB-19 ont été relevés par relecture du code le 11 novembre
(aucune reproduction : il faudrait des jours de calcul pour le premier ; le second se lit dans `src/heritage/su2_hb.rs`, fonction
`sample_x0`, et `noyaux/su2_ref64.cu`).

Vérifier ses anciens runs : voir `docs/NOTE_0.1.1.md`.

**Variable d'essai** : le binaire contient `KVU_INJECTION_FAUTE_GPU=<point>[:<rang>]` (`src/gpu/sante.rs`), qui fait échouer le
`<rang>`-ième appel GPU d'un point donné pour tester la détection de fautes. Elle ne peut que faire ÉCHOUER un run, jamais en
altérer le résultat ; ne pas la poser en production. Elle sera retirée des binaires de production en 0.2.
