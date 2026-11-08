## GATES v4-T2 — avenant WP12a-1 (implémenteur de WP12a, Opus), 2026-10-26, gelé AVANT tout run de verdict de WP12a

**Nature** : précisions et un changement de durée, rendus nécessaires par des bancs d'exploration de réglage (gamer et acemagic, `scratch/wp12a/explo_*.log`,
`gamer_explo*.log` ; sorties qui n'entrent dans aucun verdict). Il ne change aucun seuil numérique de §1.1 sauf la durée τ de FIB-ORDRE. Il est relu par
GPT-6.1-Sol (lecture seule) avant son gel, et c'est le dossier `PREVOL.md` qui fixe tout le reste.

1. **FIB-ORDRE** : réseau **L = 6, β = 8** (la gate ne fixait pas le réseau) ; durée fixée **τ = 0,08** (unités de l'intégrateur de `kvu fibre_contrainte` :
   RATTLE à impulsions N(0, 1) et M = 1, `dt` est le pas de ce temps) **au lieu de τ = 1** ; dt ∈ {τ/8, τ/16, τ/32, τ/64}, 50 départs équilibrés, impulsions
   communes, pente ∈ [1,8 ; 2,2] et mutant Euler symplectique ∈ [0,8 ; 1,2] inchangés. Raison, mesurée avant gel : sur ce réseau, τ = 1 donnerait dt = 0,125,
   où le pas RATTLE n'est plus dans le régime stable (⟨ΔH⟩ ≈ 37 à dt = 0,12 et β = 2,8, échecs de Newton) ; à β = 2,8 et τ = 0,08 le régime asymptotique
   n'est pas atteint (pente 0,04 sur 38 départs, 1,31 sur 12 départs : ΔH est dominé par quelques départs proches de la coupure du logarithme) ; à β = 8 et
   τ = 0,08 la pente vaut 2,003 (12 départs, exploration), et le mutant Euler donne 0,9. Moins de 40 départs valides ⇒ `SKIP PUISSANCE`. Le sous-ensemble de départs
   est **commun** à l'intégrateur nominal et au mutant ; seuls les départs sans solution de Newton sont écartés (comptés et rapportés), toute autre erreur est un échec.
2. **FIB-CREUTZ** : la gate compare ⟨e^{−ΔH}⟩ à 1 ; ΔH n'est défini que pour les trajectoires dont les pas et les contrôles de réversibilité aboutissent. La
   statistique est la moyenne de e^{−ΔH} **sur les trajectoires dont ΔH est défini** ; elle vaut 1 pour une involution qui préserve la mesure, restreinte à son
   domaine (Φ(D) = D : ∫_D e^{−ΔH} π = π(Φ(D)) = π(D)). Les échecs de calcul (Newton, réversibilité, évaluation) sont comptés et rapportés par cause ; au-delà de
   5 % des trajectoires, ou si ⟨ΔH⟩ > 0,5, `SKIP PUISSANCE`. Seuil, blocs et erreur : inchangés (§1.0). Le mutant Euler symplectique (hors involution, sans contrôle de réversibilité)
   doit sortir à |t| ≥ 2,7079, sinon `ECHEC_OUTIL` ; sa violation est attendue de la non-réversibilité du schéma (pas de l'ordre) et mesurée en exploration (−0,966 ± 0,005).
3. **FIB-LOGDET** : « ≤ 1e-6 relatif » s'entend **par direction** : |∇Φ·v − FD| / |FD| ≤ 1e-6 pour chacune des 20 directions (FD = Richardson à trois niveaux) ; le
   mutant est jugé par la même métrique contre le même FD (le dénominateur ne dépend jamais du gradient mutant) ; le mutant retenu est celui où les trois familles de
   dérivées (∂R, ∂L_g, ∂E_g) sont retirées ensemble ; les trois mutants partiels sont rapportés (INFO) et ne sont pas exigés (exploration : relatif à ‖∇Φ‖, ∂R seul 8e-4).
4. **FIB-REV** : les 10 points de départ sont les premiers départs équilibrés dont la trajectoire aller admet une solution de Newton ; un départ remplacé est compté.
5. **FIB-JOUET route A** : la forme de l'ajustement est contrôlée comme celle de FIB-AB (§1.0) : résidu du point du milieu par jackknife (blocs communs), |t| < 3,7207 (même
   famille m = 16) sinon `SKIP INDETERMINE` ; l'intercept est jugé par son erreur jackknife (Student t₃₉), pas par la propagation analytique.
6. **FIB-CONTR** : trajectoires de 2 pas (1000 par cellule) ; au moins un état accepté exigé (sinon `SKIP PUISSANCE`).
7. **FIB-STAT-CALIB** couvre aussi les statistiques à deux séries (différence de deux séries de même τ, m = 3 ; différence asymétrique entre une série de τ_int = τ et une série de τ_int = max(τ/10 ; 0,5), m = 3) et l'ajustement à trois points (intercept et résidu du milieu, m = 16).
8. **FIB-STAT-CALIB** : la calibration est faite à blocs ≈ 15 τ (cas le plus défavorable admis par « blocs ≥ 10 τ ») et ≈ 60 τ pour τ_int ∈ {1 ; 10 ; 50} ; les campagnes de WP12a
   ont des blocs de 160 τ à plus de 1000 τ (plus favorables). Elle s'exécute **avant** les campagnes ; son échec arrête le lot.
9. **FIB-CREUTZ** : la thermalisation (20 %) est appliquée aux indices de propositions avant tout diagnostic (⟨ΔH⟩, échecs, moyenne exponentielle portent sur le même ensemble de propositions).

---

## GATES v4-T2 — avenant WP12a-2 (implémenteur de WP12a, Opus), 2026-10-26, gelé AVANT tout run de campagne de WP12a

**Nature** : application du remède prévu par GATES §1.0 (« fausse alarme empirique ≤ 2 α nominal pour chaque seuil utilisé, sinon les seuils sont remplacés par les quantiles
empiriques de la calibration, par avenant avant les runs de campagne »). Le premier run de FIB-STAT-CALIB (commit 6dae398, graine 26102612990, 10⁴ répliques par cellule ; sortie
`scratch/wp12a/verdict_n1_stat_calib.log`) est **FAIL** avec les seuils nominaux de Student : moyenne m = 16 : 1,33·10⁻³ > 1,25·10⁻³ (τ_int = 1, blocs ≈ 15 τ) ; résidu du milieu
m = 16 : 1,39·10⁻³ et 1,38·10⁻³ > 1,25·10⁻³ (τ_int = 10 et 50, blocs ≈ 15 τ). Aucune campagne n'avait été lancée (le script s'arrête à l'échec) ; **aucune donnée de campagne n'a été vue**.
1. **Seuils retenus** = maximum, sur les six cellules (τ_int ∈ {1 ; 10 ; 50} × blocs ≈ 15 τ et ≈ 60 τ), du quantile empirique (1 − α) de |t|, calculé par le test `derivation_seuils_empiriques` avec
   **10⁵ répliques par cellule et une graine de dérivation indépendante (26102612992)**, arrondi par excès à 0,01 (sortie `scratch/wp12a/derivation_seuils.log`) : moyenne m = 1 : **2,82** (nominal 2,7079) ;
   m = 3 : 3,23 (3,1269) ; m = 4 : **3,36** (3,2322) ; m = 16 : **3,88** (3,7207) ; différence de même τ (m = 3) : 3,14 ; différence asymétrique (τ, τ/10) (m = 3) : **3,20** ; intercept d'ajustement
   (m = 16) : **3,91** ; résidu du milieu (m = 16) : **3,91**. Les seuils en gras sont ceux que les gates de WP12a utilisent : FIB-CREUTZ (2,82), FIB-RATTLE-POT (3,36), FIB-JOUET route B (3,88), FIB-JOUET route A
   (intercept 3,91, forme 3,91), FIB-A-LIBRE (3,20) ; ils remplacent partout les valeurs de §1.0 (constantes `SEUILS` de `stat.rs`).
2. **FIB-STAT-CALIB de vérification** (graine 26102612990, indépendante de la dérivation) se rejoue avec ces seuils : fausse alarme ≤ 2 α_nominal pour chaque statistique et chaque cellule. La
   **validité** des répliques (≥ 90 % de statuts verts) est exigée à blocs ≈ 60 τ (les campagnes ont des blocs ≥ 160 τ) ; à blocs ≈ 15 τ elle est rapportée sans être exigée (mesurée : 86,5-87 % à τ_int = 10 et 50, dû aux contrôles
   de stationnarité sur des blocs courts).
3. **Domaine de validité des verdicts de campagne** : tout verdict statistique de WP12a (FIB-RATTLE-POT, FIB-JOUET, FIB-A-LIBRE, FIB-CREUTZ) exige une analyse verte ET des blocs d'au moins **40 τ_int**
   (`domaine_campagne`), sinon `SKIP PUISSANCE` : la calibration couvre les blocs de 15 τ à 60 τ (validité ≥ 98 % à 60 τ) et le seuil empirique est le maximum des deux ; les campagnes prévues ont des blocs
   de 160 τ à plus de 1000 τ. Chaque cellule de calibration doit en outre fournir **au moins 5000 répliques valides** (`FIB-STAT-CALIB-EFFECTIF`), indépendamment de l'exemption de la validité à 15 τ.
4. Les gates sont des **constats de désaccord** : une valeur |t| ne peut qu'être comparée au seuil empirique ; les mutants (|t| ≫ seuil : 36 à 275 en exploration) ne sont pas affectés.
