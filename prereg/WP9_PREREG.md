# WP9 — PREREG (version finale, GELÉE) : campagne pilote « couplage effectif, un pas de bloc » sur les ensembles existants

**Statut : GELÉ avant toute donnée de la campagne (2026-10-25, ~23h55 ; GATES.md v3.5 sha256 3fe787b5803f96ea…).** GO de Kévin le 25/10/2026 (`DECISIONS_KEVIN_25-10_ADDENDUM.md`). Version finale du
projet `briefs/WP9_PREREG_projet.md` (sha256 17478caa…), écrite par le responsable WP9 (Opus). Les seuils sont reportés dans le bloc
**v3.5** de `GATES.md`. Ce fichier et `GATES.md` sont hachés au journal `journal_wp9.log` **avant** le premier bloc d'une configuration
de la campagne. Les règles de décision (§6) ont été relues par GPT-6.1-Sol avant le gel ; sa réponse et ce qui en a été retenu sont en §11.
Ce WP produit une **mesure**, pas un verdict de physique : la lecture appartient au chantier vision.

## 1. Question
Sur des ensembles SU(2) Wilson 4D **déjà générés**, que valent, après **un** pas de bloc de Bałaban L_b = 3 (géométrie `3centre`,
définition S1), les couplages effectifs β_eff donnés par trois estimateurs (`sd` : Schwinger–Dyson à un lien ; `plaq` : appariement de
plaquette ; `mcrg2` : plaquette 1×1 et rectangle 1×2), la quantité **a₁(β) = β_eff − β − 0,4081** (0,4081 = 11 C_A² ln 3/(12π²), SU(2),
ARCHITECTURE §4.3 ; convention reprise telle quelle de X3:145 / S1, non rediscutée ici), et la fraction de boucles hors domaine ?
Aucun critère ne porte sur la valeur de a₁ : c'est la quantité mesurée.

## 2. Préalables
WP5-WP8 fusionnés dans `main` 81521a3 (T1-INTEG vert sur les deux machines, ETAT_T1.md). Le binaire de la campagne est celui de la
branche `wp9` (extensions §4.4) ; les gates instrument de R1 (§6) sont rejouées **sur ce binaire** (acemagic HIP, gamer CUDA) avant toute
lecture ; sha256 consignés au journal.

## 3. Données (lecture seule ; rien n'est supprimé, déplacé ni écrit dans les ensembles)
Cellule = toutes les chaînes d'un même ensemble, d'un même L et d'un même β (définition WP8 : sommes poolées).
| famille | chemin | L | β | chaînes × configurations (espacement) |
|---|---|---|---|---|
| grandL | `chantiers/wilson_gpu_grandL_20260926/ensembles/production` | 12 | 2,3 ; 2,5 ; 2,8 ; 4 ; 8 ; 16 | 3 × 20 (`cfg_000100` … `cfg_002000`, 100 pas) |
| grandL | idem | 24 | 2,3 ; 2,5 ; 2,8 ; 4 ; 8 ; 16 | **2 × 20** (c1, c2 ; les dossiers `L24_*_c3` n'ont aucune configuration : `dump_every = 0`) |
| ym4 | `chantiers/ym4_mass_gap_0510/ens` | 24 | 1,8 ; 2,0 ; 2,15 ; 2,3 ; 2,45 ; 2,6 ; 2,8 ; 3,0 | 2 × 60 (s31, s32 ; 10 pas) |
| ym4-A | idem (`L24_b2.3_s11` … `s14`) | 24 | 2,3 | 4 × 150 (10 pas) — cellule SÉPARÉE de ym4 β = 2,3 (autre lot de graines) |
| P6 | `chantiers/vision_20261025/calculs/P6_atlas_grands_champs/ensembles_b3` | 12 | 3,0 | 3 × 20 (100 pas) |
| P6 | idem | 24 | 2,4 ; 2,6 ; 3,0 | 2 × 20 (100 pas) |
Total : 420 configurations L = 12, 1 920 configurations L = 24 (2 340), ≈ 167 Go lus. `cfg_000000.npy` (état initial) n'est jamais lu.
Manifeste sha256 de toutes les entrées : `campagnes/wp9/manifestes/entrees.sha256` (calculé sur acemagic avant les runs).
**Aucune exclusion** de configuration n'est décidée ; une configuration illisible arrête le job (pas de saut silencieux) ; une cellule
qu'on ne peut pas achever est rapportée « manquante ».

**Déjà vu (honnêteté).** Les cellules grandL L = 12 ont été bloquées et estimées par WP8 (`validation/rg_flot/flot_L12.json`) : `sd` vu
pour les 6 β ; `plaq`/`mcrg2` vus à 2,8 ; 4 ; 8 avec la courbe L = 4 de WP8, « hors étalonnage » à 2,3 ; 2,5 ; 16 ; domaine HORS partout
sauf β = 16 (f = 3,9e-4). La configuration `L24_b8_c1/cfg_002000.npy` a servi à RG-GPU (f = 0,025 rapporté par WP6/WP7). La grille
d'étalonnage de WP9 (§4.3) est étendue aux deux bouts **en sachant** que WP8 avait des cellules hors de la plage. L'information
nouvelle de WP9 : L = 24 (grandL, ym4, P6), P6 L = 12, `plaq`/`mcrg2` de L = 12 à 2,3 ; 2,5 ; 16, erreurs par blocs, réplicats.

## 4. Protocole
### 4.1 Bloc
`kvu rg flot --schema balaban --pas 1` avec **bloc sur GPU** (`--backend hip` sur acemagic, `--backend cuda` sur le gamer ; RG-GPU :
GPU = CPU à ≤ 1e-12, WP7) : 12 → 4, 24 → 8. Statistiques θ (RG-DOMAINE) de chaque configuration (`--detail-configs`).
### 4.2 Estimateurs
`--estimateurs sd,plaq,mcrg2` ; `sd` en mode exact (fraction continue), bissection sur [0, 50] ; `plaq` et `mcrg2` par inversion PCHIP
de la courbe d'étalonnage du **même côté grossier** (L = 4 pour 12 → 4, L = 8 pour 24 → 8) ; jamais d'extrapolation.
### 4.3 Courbes d'étalonnage (gelées ici : grille, graines, chaînes)
`kvu rg etalonner --protocole wp9 --L 4|8` : grille de **31 β** {0,25 ; 0,5 ; 0,75 ; 1 ; 1,25 ; 1,5 ; 1,8 ; 2 ; 2,2 ; 2,4 ; 2,6 ; 2,8 ; 3 ;
3,5 ; 4 ; 5 ; 6 ; 7 ; 8 ; 10 ; 12 ; 14 ; 16 ; 20 ; 24 ; 28 ; 32 ; 40 ; 48 ; 56 ; 64}, 3 chaînes par point, départ chaud, 2 000 pas de
thermalisation, **20 000 mesures (L = 4) et 10 000 (L = 8)**, pas = 1 bain thermique + 4 OR, moteur `CpuSu2`, graine = base + 100·i + c
(base 26 102 620 000 pour L = 4, 26 102 624 000 pour L = 8 ; c = 1, 2, 3 ; c = 9 pour la chaîne indépendante de RG-PLAQ-WP9) ; mutant
(chaîne indépendante simulée à 1,05 β) à L = 4 ; **chaînes « milieux »** (validation de l'interpolation, RG-PLAQ-WP9-MILIEU) : une chaîne
(c = 8) au milieu β_i+½ = (β_i + β_i+1)/2 de chacun des 30 intervalles, graine = base + 100·i + 8, même thermalisation et mêmes mesures. Courbe = moyenne des chaînes pondérée 1/σ², erreur de chaîne √(2τ var/N) (Sokal c = 6).
La courbe L = 4 de WP8 n'est pas utilisée pour les valeurs (elle sert au contrôle de recouvrement, INFO).
### 4.4 Extensions du code (branche wp9, déléguées à GPT-6.1-Sol puis relues et testées)
`rg flot --backend` (bloc GPU, un appareil ouvert par processus, verrou de la machine), `--bins K` (jackknife par blocs, en plus du
jackknife par chaîne inchangé), `--detail-configs` (valeurs par configuration) ; `rg etalonner --protocole wp9` ; `rg etalonnage-gates`
(RG-PLAQ appliquée aux courbes wp9). **Sans ces options, les sorties sont identiques à l'octet à celles de `main`** (gate WP9-FLOT-INCHANGE).
### 4.5 Erreurs
- **Erreur statistique principale = jackknife par blocs** de configurations consécutives, à deux tailles K et 2K : K = 4 (grandL, P6 :
  400 pas) et K = 10 (ym4, ym4-A : 100 pas) ; groupes = blocs de toutes les chaînes de la cellule (taille K : G = 15 grandL L12, 10 grandL
  L24, 12 ym4, 60 ym4-A, 15 P6 L12, 10 P6 L24 ; taille 2K : 6, 4, 6, 28 (reste rattaché au dernier bloc), 6, 4) ; estimateur non linéaire
  recalculé à chaque réplicat. **σ_stat = max(σ_K, σ_2K)**, ν = G − 1 du regroupement retenu (règle §6, contrôle des fluctuations lentes).
- Jackknife par chaîne (WP8) rapporté à côté (INFO ; 2 ou 3 réplicats seulement : non utilisé pour décider).
- Erreur d'étalonnage (plaq, mcrg2) propagée par la pente, en quadrature ; traitée comme connue (ν = ∞).
- τ_int (Sokal c = 6) de la plaquette de production et de la série bloquée rapportés ; contrôle INFO : 2 τ_int(série bloquée, en
  configurations) ≤ K.
- Systématique de schéma = écart entre estimateurs (et |β′_R − β′_P| de mcrg2), rapportée séparément, jamais un seuil.
- Systématique d'interpolation de la courbe = borne « laisser-un-de-côté » (RG-PLAQ-WP9-INTERP), rapportée (INFO).

## 5. Exécution et machines (optimisation demandée par Kévin)
- **Bloc sur GPU, les deux machines en parallèle**, partage des cellules entières (le jackknife reste dans un processus) :
  - acemagic (HIP) : grandL L = 12 et L = 24, ym4-A, P6 (≈ 1 080 configurations L = 24) ;
  - gamer (CUDA, entrées copiées par rsync et vérifiées par sha256) : ym4 (16 chaînes, 960 configurations) ;
  - **recoupement croisé** (R6) : grandL L12 β 8 et grandL L24 β 16 aussi sur le gamer, ym4 β 3,0 aussi sur acemagic.
- GPU en **`gpu_partage`** (jobs par cellule, petits réseaux L = 12/24 ; verrou partagé, 3 places sur acemagic, 4 sur le gamer) ;
  ≥ 30 Go libres sur acemagic et ≥ 24 Go sur le gamer avant chaque lancement ; fenêtres ≤ 10 min par job.
- **Courbes d'étalonnage** sur le CPU du gamer (18 fils, `gamer_run`/`gamer_lot`), puis **recertifiées** : la courbe est un calcul
  déterministe (graines gelées) ; 3 chaînes de chaque courbe (β = 0,25, 4 et 64, c = 1) sont rejouées sur acemagic et doivent avoir la
  même empreinte de séries (`empreinte_series`) : gate WP9-COURBE-REJEU.
- **Analyse** (`validation/wp9/analyse_wp9.py`) sur le gamer et rejouée sur acemagic : sorties numériques identiques à ≤ 1e-12.
- `ou_calculer` à chaque lancement ; `OMP_NUM_THREADS`/`--fils` plafonnés ; tout job > 2 Go par `calc_borne` ou `gamer_run`.
- Le temps est mesuré sur 2-3 configurations **après** le gel, puis le lancement est dimensionné (consigné au journal).

## 6. Règles de décision (GELÉES ; seuils recopiés dans GATES v3.5)
Notations. Pour une cellule c et un estimateur e : β_eff (valeur centrale, calculée sur toutes les configurations), σ_K et σ_2K (jackknife
par blocs de K et de 2K configurations), σ_cal (étalonnage). **Erreur statistique retenue σ_stat = max(σ_K, σ_2K)**, avec
ν = G − 1 du regroupement retenu (contrôle des fluctuations lentes ; le jackknife par chaîne est rapporté, INFO). Erreur d'étalonnage :
**covariance complète C_cal** entre toutes les cellules qui partagent une courbe, obtenue par rééchantillonnage paramétrique des nœuds de
la courbe (2 000 répliques, graine 26102699001 ; P et R d'un même nœud tirés avec la même déviée normale, ρ = 1, conservateur pour la
moyenne de mcrg2), PCHIP et inversion refaits à chaque réplique. ν_cal = ∞ (justification : l'erreur d'un nœud vient de 3 chaînes de
10 000-20 000 pas ; l'incertitude relative de la fenêtre de Sokal √(2(2W+1)/N) ≤ 9 % pour W ≤ 20 ⇒ ν_cal ≳ 180, à comparer à ν ≤ 59).
Covariance d'un vecteur de quantités : V = diag(σ_stat²) + C_cal. Les p-values des tests R4 et R5 sont **calibrées par Monte-Carlo sous
l'hypothèse nulle** (200 000 tirages, graine 26102699002) : quantités tirées selon N(0, V), variances jackknife tirées selon
σ_stat² χ²_ν/ν, statistique recalculée exactement comme sur les données ; p = fraction des tirages au moins aussi extrêmes. La conversion
t → z (z = Φ⁻¹(F_t(t ; ν))) et les ddl de Welch–Satterthwaite sont rapportés à titre indicatif seulement.

### R1 — Instrument (préalable ; une gate FAIL ⇒ ECHEC_OUTIL, aucune valeur rapportée comme mesure)
Sur le binaire de la campagne (sha256 consigné ; acemagic HIP et gamer CUDA) :
- `rg gates` : 17 PASS ; `rg flot-gates` : 14 PASS, lignes identiques à celles de WP8 hors durée ; `gates_rg_gpu.sh` HIP et CUDA :
  toutes PASS (RG-GPU ≤ 1e-12, RG-DOMAINE-GPU, RG-GPU-MUT ROUGE, G-DET, ENLIGNE-1) ;
- **WP9-FLOT-INCHANGE** : `rg flot` sans les options nouvelles sur grandL L12 (6 β, courbe L = 4 de WP8) = `validation/rg_flot/flot_L12.json`
  à l'octet hors champ `version` ;
- **WP9-FLOT-GPU** (HIP et CUDA) : `rg flot --backend gpu` contre `--backend cpu` sur grandL L12 β 8 : toute valeur d'estimateur à
  ≤ 1e-9 en relatif, compteurs de domaine identiques, θ max à ≤ 1e-12 ;
- **WP9-BINS** : tests (blocs = chaînes ⇒ répliques identiques au jackknife par chaîne ; moyenne iid ⇒ s/√n à 1e-12) ;
- **RG-PLAQ-WP9** (L = 4 et L = 8, P et R) : chaîne indépendante (c = 9) à chacun des 31 β : z en β ; PASS ssi max|z| < **3,59**
  (Bonferroni, α = 0,01 sur 31 points : Φ⁻¹(1 − 0,01/62)) et χ² p > 0,01 (survie exacte, 31 ddl) ; **mutant** 1,05 β à L = 4 : ROUGE exigé ;
- **RG-PLAQ-WP9-MILIEU** (L = 4 et L = 8, P et R) : une chaîne indépendante (c = 8) au milieu de chacun des 30 intervalles de la grille,
  comparée à la valeur PCHIP de la courbe : z = (P_mes − P_PCHIP)/hypot(σ_mes, σ_courbe) ; PASS ssi max|z| < 3,59 et χ² p > 0,01 (30 ddl) ;
  c'est la validation de l'interpolation ENTRE les nœuds ;
- **WP9-ANALYSE-INV** : l'inversion PCHIP refaite en Python par le script d'analyse redonne les β_eff du binaire à ≤ 1e-8 en relatif
  (cellules dans la plage) ;
- **WP9-COURBE-REJEU** : ≥ 3 chaînes de chaque courbe (L = 4, L = 8 : β = 0,25, 4 et 64, c = 1) rejouées sur acemagic : empreintes des
  séries identiques à celles du gamer.

### R2 — Domaine (étiquette par cellule)
Étiquette RG-DOMAINE de v3.3 sur la réunion des boucles de toutes les configurations de la cellule : **DANS ssi f ≤ 1e-3 et aucune branche
ambiguë**, sinon HORS. Cellule HORS ⇒ ses a₁ sont publiés à part et **ne servent à aucune conclusion de physique**. Rapportés aussi :
f par configuration (min, médiane, max), fraction des configurations individuellement HORS.

### R3 — Hors étalonnage, non calculable
- Valeur centrale hors de [0,25 ; 64] (plaq, mcrg2) ⇒ borne « < 0,25 » ou « > 64 », jamais extrapolée ; sd > 50 (borne de la bissection) ⇒
  « > 50 ». Une borne n'entre dans aucun ajustement.
- Valeur centrale dans la plage mais **un réplicat** jackknife hors plage (σ_K ou σ_2K indéfini), ou **> 1 % des répliques** d'étalonnage
  hors plage ⇒ « non calculable » : valeur rapportée, exclue des ajustements, jamais supprimée en silence.

### R4 — Localité L = 12 contre L = 24 (par estimateur ; par AJUSTEMENT, jamais un point seul)
Paires au même β dans la même famille : grandL β ∈ {2,3 ; 2,5 ; 2,8 ; 4 ; 8 ; 16} ; P6 β = 3,0. Pour chaque paire valide (deux valeurs
calculables) : Δ_β = β_eff(12) − β_eff(24) (= Δa₁). Vecteur Δ sur l'ensemble B des paires valides (n = |B|), covariance
V_Δ = diag(σ_stat(12)² + σ_stat(24)²) + C_cal,4 + C_cal,8 (cellules indépendantes ; corrélations d'étalonnage complètes).
- Modèle nul Δ ≡ 0 : χ²₀ = Δᵀ V_Δ⁻¹ Δ, p₀ (MC sous le nul ; indicatif : χ² à n ddl).
- Modèle constant Δ ≡ δ (moindres carrés généralisés) : δ̂ = 1ᵀV⁻¹Δ / 1ᵀV⁻¹1, σ_δ = (1ᵀV⁻¹1)^(−1/2), p_δ bilatéral (MC sous le nul) ;
  χ²₁ (n − 1 ddl) rapporté.
- Verdict par estimateur, n ≥ 2 :
  - **EFFET_DE_VOLUME** si p₀ < 0,00135 **ou** p_δ < 0,00135 (budget 0,0027 ≈ « 3σ » partagé par Bonferroni entre les deux tests) ;
  - **COMPATIBLE_LOCALITE « à cette précision »** si p₀ ≥ 0,0455 **et** p_δ ≥ 0,0455, avec la borne |δ| < |δ̂| + t₀,₉₇₅;ν_δ σ_δ (non-rejet,
    pas une équivalence ; la borne ne borne pas chaque Δ_β si le modèle constant est inadéquat : χ²₁ rapporté) ;
  - **INDETERMINE** sinon ; n < 2 ⇒ **NON_TESTABLE** (les Δ sont rapportés comme mesures). Attendu : au plus 2 paires DANS/DANS
    (grandL β = 16, P6 β = 3,0 si DANS), donc NON_TESTABLE est une issue normale.
- B est formé séparément (i) des paires **DANS/DANS** (seul ensemble qui porte une conclusion) et (ii) de toutes les autres paires
  (publié à part, descriptif, sans conclusion). Trois estimateurs ⇒ trois verdicts séparés ; une lecture « au moins une alerte sur trois »
  doit compter la multiplicité (≈ ×3).

### R5 — Réplicats (contrôle de la statistique ; gate WP9-REPLIQUE, par estimateur)
Cellules indépendantes de même (L = 24, β) : β = 2,3 {grandL, ym4, ym4-A} ; 2,6 {ym4, P6} ; 2,8 {grandL, ym4} ; 3,0 {ym4, P6}.
Contrastes orthogonaux (paire : différence ; triplet : différence des deux premiers, puis le troisième contre leur moyenne pondérée), erreurs
σ_stat seulement (même courbe L = 8 pour toutes les cellules L = 24 : l'erreur d'étalonnage s'annule au premier ordre à cible voisine ; la
part résiduelle est incluse par la covariance C_cal des contrastes) ; χ²_R = cᵀ V_c⁻¹ c sur les contrastes calculables (≤ 5) ;
p_R par MC sous le nul. **PASS (« aucune anomalie détectée ») ssi p_R ≥ 0,001** ; FAIL ⇒ erreurs de l'estimateur déclarées non calibrées et
ses verdicts R4 deviennent INDETERMINE (erreurs non calibrées). Moins de 2 contrastes ⇒ SKIP rapporté (verdict R4 mentionné « erreurs non
contrôlées par réplicats »). Rapportés : facteur de surdispersion s = √(χ²_R/ddl) et son intervalle à 95 % ; **puissance annoncée** : avec
5 contrastes, des erreurs sous-estimées d'un facteur 2 ne sont détectées qu'≈ 40 % du temps (R5 est un détecteur, pas une certification).
Ces cellules sont HORS domaine : elles contrôlent la statistique, elles ne portent aucune conclusion de physique.

### R6 — Deux machines (gate WP9-2MACH)
Cellules calculées sur les deux GPU : grandL L12 β 8 et grandL L24 β 16 (primaires acemagic), ym4 L24 β 3,0 (primaire gamer). PASS ssi
toute valeur d'estimateur et d'erreur à ≤ 1e-9 en relatif, compteurs de domaine (boucles hors, ambiguës) identiques, θ max à ≤ 1e-12.
FAIL ⇒ les cellules du gamer sont recalculées sur acemagic et l'incident est rapporté.

### R7 — Rapporté sans critère
Table a₁(β) par estimateur, par L et par famille, avec σ_stat, σ_cal et erreur totale ; f et étiquette par cellule ; écart de schéma
(étendue des trois estimateurs ; |β′_R − β′_P| de mcrg2) ; ajustement descriptif β_eff = Z β + c (moindres carrés généralisés) sur grandL
β ∈ {4 ; 8 ; 16} par (estimateur, L) — cellules 4 et 8 HORS : descriptif ; τ_int ; rapports σ_2K/σ_K et σ_chaîne/σ_K (INFO).
**Estimateur de tête** pour la phrase de résultat : `sd` (sans courbe ; validé par RG-SD) — ce n'est pas « le » couplage.

## 7. Sorties
`U/campagnes/wp9/` (≤ 1 Go) : `flot/<machine>/<cellule>/flot.json`, `etalonnages/*.json`, `manifestes/` (entrées, binaires, commandes),
`analyse/` (tables JSON et Markdown), `VERDICT.md` (format vision : phrase de résultat, mesure, limites). Branche `wp9` : code, tests,
`prereg/WP9_PREREG.md`, `validation/wp9/` (script d'analyse, courbes, verdicts de gates).

## 8. Coût (ESTIMATION, à remplacer par la mesure)
Bloc GPU ≈ 0,2-0,3 s par configuration L = 24 (WP7 : HIP 0,23-0,27 s, CUDA 0,16-0,18 s), négligeable à L = 12 ; lecture ≈ 85 Mo par
configuration L = 24 ⇒ ≈ 10-15 min de GPU par machine. Courbes (L = 8 : 93 + 31 + 30 chaînes de 12 000 pas ; L = 4 : 93 + 31 + 31 + 30
chaînes de 22 000 pas) : ≈ 9-10 h-fil ⇒ ≈ 35 min sur 18 fils du gamer. Analyse (rééchantillonnage, MC sous le nul) : quelques minutes.

## 9. Ce que WP9 ne dit pas
Pas de flot à plusieurs pas, pas de limite continue, pas de verdict sur le mur θ ni sur le gap. Les cellules HORS domaine (attendu :
presque toutes, sauf β = 16) ne servent à aucune conclusion. a₁ dépend du schéma (trois estimateurs, aucun n'est « le » couplage).

## 10. Journal
`U/journal_wp9.log` (lignes `WP9`) : gel (sha256 de ce fichier et de GATES.md), runs, temps, incidents, verdicts de gates.

## 11. Relecture des règles par GPT-6.1-Sol (avant le gel)
Commande : `codex exec --skip-git-repo-check -m gpt-6.1-sol -c model_reasoning_effort=high -s read-only "<règles + question>"` sur le
brouillon `scratch/wp9/regles_decision_brouillon.md` (sha256 consigné au journal) ; réponse intégrale : `scratch/wp9/codex_relecture_regles.txt`
(copiée dans `campagnes/wp9/manifestes/`). Verdict du relecteur : « pas entièrement calibrées ». Défauts et suite donnée :
| # | défaut relevé (cas plausible) | suite dans cette version |
|---|---|---|
| 1 | covariances d'étalonnage entre β (nœuds PCHIP partagés) et entre P et R de mcrg2 ignorées (ρ = 0,8 ⇒ faux « 3σ » à 2,5 % au lieu de 0,27 %) | **adopté** : C_cal complète par rééchantillonnage des courbes (PCHIP + inversion refaits), P/R tirés avec ρ = 1 ; moindres carrés généralisés |
| 2 | fluctuations lentes non vues par τ de la plaquette fine ; ν de Welch approximatifs ; ν_cal = ∞ à justifier | **adopté** : σ_stat = max(σ_K, σ_2K) ; p-values par MC sous le nul (la loi des variances estimées et la règle du max y sont reproduites) ; ν_cal justifié (§6) |
| 3 | interpolation non validée entre les nœuds ; réplicats hors grille près des bords | **adopté** : RG-PLAQ-WP9-MILIEU (30 chaînes indépendantes aux milieux, par courbe) ; règle « non calculable » (R3), jamais de suppression silencieuse |
| 4 | multiplicités : union des deux tests de R4 (0,43 % au lieu de 0,27 %) ; 31 contrôles \|z\| < 3 dans R1 (≈ 8 % de rejet à tort par courbe) | **adopté** : budget 0,0027 partagé (0,00135 par test) en R4 ; seuil Bonferroni 3,59 (α = 0,01) pour max\|z\| des courbes ; multiplicité entre estimateurs déclarée |
| 5 | R5 peu puissant (facteur 2 détecté ≈ 40 %) ; contrastes studentisés non indépendants | **adopté** : PASS = « aucune anomalie détectée » ; surdispersion et intervalle rapportés ; puissance annoncée ; p_R par MC sous le nul |
| 6 | « compatible » sans contrôle des faux négatifs ; borne 2σ mal calibrée à petits ν | **adopté** : interprétation « non-rejet à cette précision » écrite ; borne t₀,₉₇₅;ν_δ ; **non adopté** : test d'équivalence à tolérance ε (aucune tolérance physique justifiable a priori pour a₁) |
Conservé (validé par le relecteur) : conversion signée t → z (indicative), ddl n et n − 1, recalcul non linéaire du jackknife, absence
d'extrapolation, séparation DANS/HORS, recoupement entre machines, aucun critère sur la valeur de a₁, transparence sur le « déjà vu ».
Remarque du relecteur reprise en R4 : au plus deux paires DANS/DANS ⇒ NON_TESTABLE est une issue normale du protocole.
