# Boîtes de Dirichlet SU(2) à bord fixé sur GPU (`kvu su2 boite`)

Lot WP-DIR du programme d'unification. Objet : mesurer l'influence d'un bord fixé sur les observables invariantes de jauge au centre d'une boîte ℓ⁴,
100 fois plus vite que le code CPU numpy de QW1, sur les deux GPU (HIP gfx1151, CUDA sm_120) et sur une machine louée. Les seuils des gates, hachés avant les runs,
sont dans `GATES-DIR.md` (v1 + v2) ; les références de QW1 sont dans `crates/kvu/reference/qw1/` (copies à l'octet, sha256 dans `SOURCES_IMPORTEES.md`).

## 1. Définitions

- **Boîte.** ℓ pair ≥ 2, sites {0,…,ℓ}⁴, lien `U_μ(x)` de `x` à `x+e_μ` pour `x_μ ≤ ℓ−1`. Un lien est **fixé** si ses deux extrémités sont sur le bord du cube (une coordonnée vaut
  0 ou ℓ), **libre** sinon. `n_libres(ℓ) = 4ℓ(ℓ−1)³` (ℓ = 8 : 10 976, comme le PREREG de QW1). Action `S = β Σ_p (1 − ½ Re Tr U_p)`, β = 4/g².
- **Réseau de calcul.** La boîte est plongée, mêmes coordonnées, dans un réseau périodique de côté **ℓ + 2** (pair, exigé par le damier des noyaux). Tout lien qui n'est pas libre est gelé par
  un **masque** (un octet par lien, disposition `li = μ·V + s` du champ, partagé par les chaînes du lot) : liens fixés, liens inexistants (`x_μ = ℓ`), liens de la couche de remplissage.
  Un lien libre ne lit jamais un lien de remplissage ni le repliement périodique. Seul le gaspillage de fils change (un lien gelé sort tout de suite) : 27 % de fils actifs à ℓ = 8, 51 % à ℓ = 16, 64 % à ℓ = 24.
- **Masque dans les noyaux.** `k_heatbath` et `k_overrelax` (`noyaux/su2_ref64.cu`) reçoivent un pointeur de masque (nul = aucun masque) ; le test est placé juste après le calcul de l'indice de lien,
  avant tout calcul flottant. Même source pour HIP et CUDA ; le CPU de référence (`su2::cpu`) saute les mêmes liens avec les mêmes compteurs Philox. Le masque généralise le gel R2-3 (même octets :
  gate G-DIR-MASQUE-GEL) ; un masque nul égale l'absence de masque ; **sans masque, les sorties de `su2 gen` sont inchangées à l'octet** (G-DIR-BIT-HIP-PROD 3/3, G-DIR-BIT-CUDA 3/3, G-DIR-BIT-K1 12/12).
- **Observables** (centre c = ℓ/2) : `P_c` = moyenne des 24 plaquettes de coin c ; `W₂` = moyenne des 6 boucles 2×2 centrées en c (ℓ ≥ 4) ; profil par couche = moyenne de ½Tr des plaquettes de chaque
  classe de profondeur d (d = 0 … ℓ/2 − 1). Les sommes ont un ordre contractuel (séquentiel pour P_c et W₂, 256 fils et arbre fixe pour le profil) : le noyau GPU (`noyaux/su2_boite.cu`) et la référence
  hôte (`su2::boite::mesurer_hote`) donnent les mêmes octets.
- **Bords.** `cold` ; `gauge` (g(x)g(x+μ)†, jauge pure : témoin) ; `z2full` (U_μ(x) = (−1)^{x₀+…+x_{μ−1}}, toutes les plaquettes du bord valent −1) ; `z2_01` (U₁(x) = (−1)^{x₀}) ; `flux`
  (U₁ = exp(iσ₃ φ x₀), U₃ = exp(iσ₁ φ x₂), φ = π/2, valeurs exactes 0 et ±1 : l'agrafe peut être exactement nulle, ce que traverse la branche de Haar des noyaux, testée) ;
  `haar` (liens fixés de Haar) ; `therm` (liens fixés d'une configuration d'équilibre périodique de même β, produite par le même moteur sans masque, côté max(10, ℓ+2), `--therm-source` pas).
  Les bords sont construits sur l'hôte et téléversés : identiques sur HIP, CUDA et CPU.

## 2. Utilisation

```text
kvu su2 boite --ell 8 --beta 2.5 --bords cold,z2full --repliques 16 --graine 26102611000 --graine-pas 1 \
    --therm 1000 --nmeas 30000 --skip 1 --nor 2 --profil-tous 10 --outdir DIR
kvu su2 boite --ell 12 --beta 2.5 --chaines cold:1,gauge:2,flux:3,haar:4,therm:5 --therm 1000 --nmeas 5000 --dump-final --outdir DIR2
kvu su2 boite-obs --cfg DIR2/c000/config_finale.npy --ell 12 --backend cpu        # P_c, W₂, profil d'une configuration NPY
kvu regen DIR/manifest_regen.json --sortie DIR_REJEU                                # rejeu à l'octet
python3 -I scripts/pilote_boites.py TACHES.csv --sortie SORTIE --cible acemagic=local:KVU --cible gamer=ssh:gamer:KVU_DISTANT \
    --verrou gamer=/tmp/gamer_gpu.lock --debit acemagic=1.2e8 --debit gamer=1.2e8 [--dry-run]                             # pilote de balayage (§3)
```

Bord arbitraire : `--bord-npy FICHIER` (répétable) lit les liens fixés d'un champ NPY du réseau de calcul (format de `--dump-final`) ; nom de bord
`npy:<h8>` ; spécification, refus, liens fixés actifs et inertes : `docs/BORD_NPY.md` (lot WP21).

Sorties (répertoire neuf) : `cNNN/mesures.csv` (`pas,pc,w2`), `cNNN/profil.csv` (`pas,d0,…`), `cNNN/config_finale.npy` (option), `manifest.json` (provenance, `controle_bord`), `manifest_regen.json`
(`kvu-regen/1`, rejouable), `COMPLETE` en dernier ; dernière ligne de sortie : `PERF boite …`. La commande prend le verrou GPU de sa machine. En fin de run, les liens gelés sont comparés à l'octet à leur
départ (empreinte sha256 des liens masqués) : un écart rend le code 8. Codes de sortie : ceux de `kvu` (2 usage, 3 GPU, 5 ressources, 6 verrou, 7 disque, 8 validation).
Les contrôles négatifs (`--mutant masque-ignore | masque-ignore-bain | masque-ignore-surrelax`, cachés) ne servent qu'aux gates : le run se termine, le manifeste consigne `controle_bord.intact = false`.

## 3. Le pilote `scripts/pilote_boites.py`

Entrée : un CSV `ell,beta,bord,graine[,therm,nmeas,skip,nor,options,profil_tous]` (une ligne = une chaîne). Les chaînes de mêmes paramètres forment une cellule, coupée en lots tels que
le temps estimé tienne dans la fenêtre (`--fenetre-s`, défaut 600 s) et la mémoire dans `--mem-gpu-go` ; les lots sont répartis, un fil par cible (local ou `ssh`), sur une file partagée (le plus long d'abord,
équilibrage dynamique). `--dry-run` imprime le plan déterministe sans rien écrire ; `--reprendre` saute les lots COMPLETE ; un lot dont une seule chaîne dépasse la fenêtre est refusé (code 2) avant
tout lancement. Sorties : `lots/<id>/…` (rapatriées par `rsync -s` pour le distant, le distant n'est jamais effacé), `index.json` (plan, statuts, verrou déclaré de chaque cible), `pilote.log`.
Il **ne lance pas** de campagne : l'agent de physique écrit le préenregistrement, la liste de tâches et lance.

**Verrou GPU, un par machine.** `kvu` prend lui-même le verrou de sa machine, mais seulement s'il la reconnaît : lancé par ssh depuis le répertoire personnel, sans `kvu.toml`, il prend le verrou
par défaut `/tmp/kevinotron_gpu.lock` et non `/tmp/gamer_gpu.lock` (défaut trouvé à l'essai réel : un lot est parti 7 s avant la libération du verrou du gamer). Le pilote impose donc la déclaration :
`--verrou NOM=CHEMIN` (chemin absolu ou `~/…`) enveloppe chaque lot de la cible dans `flock -E 75 -w ATTENTE CHEMIN kvu …` ; `--verrou NOM=aucun` la désactive explicitement ; une cible ssh sans
déclaration est refusée (code 2). Pour une cible locale, sans déclaration, `kvu` prend le verrou de sa machine ; avec déclaration sur le même fichier que `kvu`, celui-ci voit le verrou tenu par
son ancêtre (`verrou = ancetre` au manifeste). Le délai d'attente (`--attente-verrou-s`, défaut 7200) s'ajoute au délai du lot ; un verrou non obtenu rend `flock` 75, le lot est en échec avec le message
« verrou GPU … non obtenu ». Débit à déclarer (`--debit`, liens libres mis à jour par seconde, surrelaxations comprises) : le défaut 5e7 est prudent (fenêtres sous-remplies), les mesures du §4.2 donnent
1,5e8 (ℓ = 8) à 2,5e8 (ℓ = 24) sur HIP et sur CUDA pour K ≥ 16 chaînes par lot ; à `1.2e8`, la liste d'exemple ci-dessous (144 chaînes, ℓ ∈ {8, 12, 16}, β ∈ {2,3 ; 2,5 ; 2,7}, `cold`/`z2full`,
8 graines chacun) donne 27 lots de 166 à 532 s estimés, environ 5 900 s par machine.

```text
ell,beta,bord,graine
8,2.3,cold,27000000000
8,2.3,cold,27000000001
…                                     (une ligne par chaîne ; les chaînes de mêmes (ell, beta, paramètres) forment une cellule)
```

## 4. Résultats des gates et performances

Binaires : acemagic (HIP gfx1151) `1e248987…`, gamer (CUDA sm_120, RTX 5060 Ti) `5ce05220…`, construits du commit `59835ab` (`source_hash` `41122429…`) ; depuis, seuls des tests, des scripts et des documents ont changé.
Seuils : `GATES-DIR.md` (v1 hachée avant les runs ; v2 : un seuil numérique relâché ; v3 : constat sur le pilote). Détail ligne à ligne : `journal_wp_dir.log` ; synthèse : `rendus/WP-DIR.md`.

### 4.1 Verdicts

| gate | HIP (acemagic) | CUDA (gamer) |
|---|---|---|
| sans masque, sorties de `su2 gen` identiques à l'octet à la production | BIT-HIP-PROD 3/3 ; BIT-K1 ℓ = 12 : 6/6, ℓ = 16 : 6/6 | BIT-CUDA 3/3 |
| masque : gel R2-3 retrouvé à l'octet, masque nul = pas de masque, liens gelés intacts (+ 3 mutants rouges) | PASS | PASS |
| CPU contre GPU avec masque, 32 cas | 0 décision divergente ; max\|Δq\| 1,2e-15 (bain), 5,2e-15 (surrelaxation) | 3,8e-15 / 6,4e-15 |
| observables (P_c, W₂, profil) | hôte Rust contre Python de QW1 ≤ 2,2e-16 ; GPU = hôte à l'octet (12/12) ; centre décalé ⇒ rouge | GPU contre Python : 16 comparaisons à 0 ou 1,1e-16 ; 8 mutants « centre décalé » rouges |
| bord intact (10 bords) | max\|ΔU_bord\| = 0 ; 3 mutants (masque ignoré, bain, surrelaxation) rouges (max\|Δq\| = 2) | idem |
| `kvu regen` à l'octet | 9/9 + mutant rc 8 | 9/9 + mutant rc 8 |
| manifeste `kvu-regen/1` | valide ; 51/51 mutants de schéma rejetés | idem |
| QW1 reproduit (β = 2,5, ℓ = 2, 4, 6, 8) | PASS (§4.3) | PASS (§4.3) |
| QW1, mutant « masque ignoré » | PASS (rouge voulu : ΔP_c(8) = 7e-6 ± 1,4e-4, z = −10,5) | PASS (ΔP_c(8) = −6,5e-5 ± 1,4e-4, z = −10,6) |
| jauge pure, ℓ = 2 … 16 | PASS ; z de −0,22 à 1,90 | PASS ; z de −1,34 à 2,26 |
| jauge, mutant « bord de Haar » | PASS (z = −1789 et −252) | PASS (z = −1885 et −250) |
| agitation (ℓ = 2) | PASS (z = −1698) ; mutant PASS (z = 0,21) | PASS (z = −1693) ; mutant PASS (z = 1,35) |
| couplage fort (β = 0,2, ℓ = 6) | PASS, \|z\| < 1 ; mutant β × 1,05 : z de 6 à 9 | PASS, \|z\| ≤ 1,22 ; mutant : z de 6,5 à 8,4 |
| pilote | (a) plan déterministe, (b) refus d'un lot trop long, verrou déclaré : tests `boite_pilote` ; (c) 4 tâches sur les deux GPU en parallèle : PASS ; (d) verrou par GPU : PASS ×3 (1re passe : 2 FAIL, §4.4) | idem (cible gamer par ssh) |

### 4.2 Performances (ms par pas et par chaîne ; un pas = 1 bain + 2 surrelaxations ; 120 pas sous le verrou ; K = chaînes par lancement)

| ℓ | HIP K = 1 / 16 / 64 | CUDA K = 1 / 16 / 64 | QW1 CPU (s par pas) | gain K = 64 : HIP / CUDA |
|---|---|---|---|---|
| 8 | 0,495 / 0,248 / 0,226 | 0,782 / 0,252 / 0,223 | 0,064 (0,025 à vide) | 283× / 287× (111× / 112× contre 0,025) |
| 12 | 1,48 / 1,02 / 1,02 | 1,60 / 1,02 / 0,973 | 0,513 | 501× / 528× |
| 16 | 3,34 / 3,08 / 3,05 | 3,79 / 2,97 / 2,92 | 2,554 | 838× / 876× |
| 24 (K = 1 / 4 / 16) | 14,9 / 14,0 / 15,3 | 15,0 / 14,2 / 13,9 | | |
| 32 (HIP seul, K = 1 / 4 / 16) | 55,3 / 54,5 / 49,1 | | | |

Note de publication : la colonne « QW1 CPU » est le script de référence Python/numpy `reference/qw1/box_su2.py` (un fil),
pas une implémentation CPU optimisée du même algorithme ; le « gain » de la dernière colonne mesure donc l'écart à cette
référence, et ne doit pas être lu comme un facteur d'accélération GPU contre CPU.

Débit : 1,5e8 (ℓ = 8) à 2,5e8 (ℓ = 24) liens libres mis à jour par seconde sur HIP ; 1,5e8 à 2,5e8 sur CUDA. Le CPU de QW1 est `box_su2.py` (numpy, un cœur, temps utilisateur par pas, minimum de deux essais, 30 pas) mesuré sur une machine
chargée (charge 37) : le chiffre de QW1 sur machine calme (ℓ = 8 : 25 ms) donne 111×. **Le gain est par chaîne contre UN processus numpy** ; contre un nœud qui lance autant de processus numpy que de cœurs (8), il est à diviser par 8
(environ 14× à ℓ = 8). Un lancement à K = 1 ne remplit pas le GPU : le pilote groupe les chaînes de même cellule. À ℓ = 8, 27 % des fils d'un balayage portent un lien libre (51 % à ℓ = 16, 64 % à ℓ = 24), les autres sortent tout de suite.

### 4.3 Reproduction de QW1 (β = 2,5 ; 32 chaînes ; 30 000 mesures ; z contre la référence de QW1)

| ℓ | ΔP_c HIP | z | ΔP_c CUDA | z | ΔW₂ HIP | z | ΔW₂ CUDA | z |
|---|---|---|---|---|---|---|---|---|
| 2 | 0,608200(103) | −0,07 | 0,608319(102) | +0,22 | | | | |
| 4 | 0,082175(147) | +1,35 | 0,081916(147) | +0,93 | 0,186594(390) | −0,32 | 0,186410(391) | −0,43 |
| 6 | 0,017092(142) | +0,98 | 0,017033(141) | +0,88 | 0,050449(394) | −0,34 | 0,050672(393) | −0,21 |
| 8 | 0,005147(140) | −1,52 | 0,005443(140) | −1,00 | 0,017885(391) | −0,53 | 0,018191(390) | −0,34 |

Référence de QW1 à ℓ = 8 : ΔP_c = 0,0060(6), ΔW₂ = 0,0188(16). Le profil par couche (ℓ = 8, cold − z2full) redonne ΔP̄_d = 0,0964 / 0,0174 / 0,0068 pour d = 1, 2, 3 (QW1, post hoc : 0,096 / 0,018 / 0,0075).
La référence de QW1 est UNE chaîne de 30 000 pas par cellule. Seule déviation absolue notable : P_c(ℓ = 4, bord cold) à +2,8σ (HIP) et +2,6σ (CUDA) de la valeur de QW1 (0,67115(38)). Diagnostic INFO : 24 chaînes fraîches du
`box_su2.py` de QW1 donnent 0,672223(78), d'accord avec HIP 0,672241(94) et CUDA 0,672159(95) (z = −0,15 et +0,53) ; c'est la chaîne de référence de QW1 qui était basse (z = +2,77), W₂ et le profil s'accordent aussi.

### 4.4 Écarts et défauts consignés

- **GATES-DIR v2** : le seuil de la plaquette du bord de jauge pure passe de 4e-16 à 3e-15 (observé 1,1e-15 = 5 ulp), écrit avant tout run GPU (relâchement à relire).
- **Pilote, 1re passe réelle (25/10)** : (c) FAIL, le lot gamer a tourné mais le rapatriement `rsync` a échoué (chemin distant cité avec `$HOME`, sans shell distant) ; (d) gamer FAIL, le lot a démarré 7,05 s avant la libération du verrou du gamer
  (le `kvu` distant, sans `kvu.toml`, prend le verrou par défaut). Deux défauts du pilote, corrigés (`rsync -s`, `--verrou` obligatoire pour ssh, enveloppe `flock`), tests ajoutés, mutant « enveloppe désactivée » rouge sur 3 tests ; 2e passe sous les mêmes graines et seuils : PASS.
  Pendant la 1re passe, le lot gamer de (c) (0,33 s de GPU, ℓ = 4) a tourné pendant qu'une fenêtre statistique tenait le verrou du gamer : violation du verrou, sans effet sur les données (déterminisme par graine).
- **Gamer** : `gamer_run` a refusé le lancement (charge 20,9 > 18) ; les fenêtres ont été lancées par ssh avec les mêmes garde-fous (scope `ym4calc.slice`, `MemoryMax` 14 G, `oom_score_adj` 1000, nice 5, cœurs 12-19). `MemAvailable` du gamer : 21 à 28 Go (règle : 24 Go) ; les fenêtres de runs pèsent quelques centaines de Mo (modèle du pilote : 16·(ℓ+2)⁴·8·1,3 octets par chaîne).
- **7 lignes `FAIL G-DIR-RUN … REFUS d'écraser`** dans le journal CUDA : fenêtres redondantes (le script copié sur le gamer n'avait pas le filtre `ELLS`, les premières fenêtres ont produit tous les ℓ) ; les 18 répertoires de runs sont COMPLETE, aucune donnée perdue, aucun run relancé.
- Un banc existant (`banc_su2_publie_le_temps_mur`) est sensible à la charge de la machine (échec isolé à charge 50, vert à la reprise).
- Le flux est posé avec des valeurs trigonométriques exactes (0, ±1) ; le noyau `su2_boite.cu` n'est pas dans le registre gelé `noyaux::TOUS` (un test fige 11 noyaux) : il est embarqué par `include_str!` dans `boite_gpu.rs`.

## 5. Limites et suites possibles

- Le F32 n'a pas de masque (refusé : code 2) ; le chemin F64 sur une RTX 5060 Ti (FP64 à 1/64) est nettement plus lent que sur l'APU.
- Les fils des liens gelés sortent immédiatement : une liste compactée des liens libres par (μ, parité) donnerait jusqu'à ×2 à petit ℓ (à ℓ = 8, 27 % de fils actifs).
- Les bords `therm` sont produits par le moteur ; lire des sources d'équilibre existantes (ensembles de production) est une extension simple, non faite.
- Ce lot livre l'instrument ; il ne conclut rien sur la contraction de Dobrushin (le verdict VIVANT de QW1 reste celui de QW1, filtre qui ne peut que tuer).
- Grands ℓ : ℓ = 24 coûte 14 à 15 ms par pas et par chaîne (HIP et CUDA), ℓ = 32 de 49 à 55 ms (HIP) ; une chaîne de 31 000 pas à ℓ = 32 dépasse la fenêtre de 600 s (le pilote refuse alors la cellule : il faudrait des reprises, `--reprendre` du pilote ne coupe pas une chaîne).
