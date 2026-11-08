# WP12a — PREREG du re-verdict « les échecs de Newton sont un coût, pas un biais » (01/11/2026, AVANT toute donnée de verdict)

Auteur : agent Sonnet 5.5 (reprise du 01/11). Relecteur : Kimi K3.1 Pro. Aucun seuil, graine ni gate gelé de WP12a n'est modifié ; le seuil t utilisé (3,88, m = 16) est le seuil empirique déjà gelé `SEUILS.moy_m16` (avenant WP12a-2). Code testé : wp12a b20cebc tel quel (la reprise n2bis y a observé 29 échecs de Newton et rendu `SKIP FIB-CONTR PUISSANCE`, seuil gelé de 1 %).

## 1. Question et définitions
Une proposition RATTLE en échec de calcul (Newton non convergé, réversibilité, évaluation) est rejetée : l'état est conservé ET compté (`rattle::chaine`). Question : cette règle biaise-t-elle l'espérance de l'estimateur, ou ne coûte-t-elle que de l'acceptation (et du temps d'autocorrélation) ?
- Estimateur de production E_rej : moyenne sur toutes les trajectoires, états répétés compris.
- Mutant E_drop : on retire de la série les entrées issues d'une proposition en échec.
- Mutant E_retry : un échec est remplacé par une nouvelle proposition (nouvelle impulsion) jusqu'au succès.
Argument attendu (exact dans le modèle idéal ; la tolérance de réversibilité 1e-9, qui ne contrôle au retour que la position, n'en donne pas une borne rigoureuse du biais stationnaire, d'où le contrôle numérique) : si l'ensemble de succès est invariant par renversement (le contrôle de réversibilité de chaque pas l'impose), le noyau est π-réversible et E_rej est sans biais ; E_drop et E_retry échantillonnent π(x)(1 − f(x)), de biais −Cov_π(O, f)/(1 − E_π f), f(x) = P(échec | x). Le contrôle ci-dessous teste ces trois énoncés numériquement.

## 2. Contrôle à vérité connue (jouet CT-7, courbe y = 1,2 sin x de ℝ², x replié dans [−π, π])
Le jouet de production non replié dérive (|x| ≈ 10³ après 10⁶ trajectoires) et l'arrondi de `a sin x` atteint la tolérance de Newton : échecs dépendant de |x|, hors hypothèse. Le contrôle utilise donc `CourbePeriodique` (même code de production, x replié à chaque avance, écart périodique). Observé en exploration (hors verdict) : avec x replié le taux d'échec est stationnaire sur 10⁶ trajectoires.
- Cellules (nb_pas = 6, max_newton = 50, tol_newton 1e-13, tol_rev 1e-9) : M = diag(1, M2), M2 ∈ {1 ; 4}, dt ∈ {0,15 ; 0,5 ; 0,8 ; 1,0} pour M2 = 1 et {0,15 ; 0,3 ; 0,4 ; 0,5} pour M2 = 4. Taux d'échec attendus de 0 à ≈ 30 %.
- N = 2 000 000 trajectoires par chaîne ; graines : cellule c, g = 26110160000 + 10c : production g+1, réessai g+2, profil d'échec g+3, invariance g+4. Observables cos x et cos 2x.
- Vérité : quadrature de CT-7 recalculée en Python (poids exp(−cos x/2 − 0,72 sin² x), 200 000 points).
- Statistique : `stat::analyser` (thermalisation 20 %, jackknife 40 blocs ≥ 10 τ_int, domaine de campagne : blocs ≥ 40 τ_int).
- Profil d'échec : 400 nœuds de x × 20 000 propositions par nœud avec la routine de production ; biais prédit −Cov_π(O, f)/(1 − E_π f), erreur Monte-Carlo du profil propagée.
- Invariance : 200 000 départs par cellule (x uniforme, impulsion de production) ; tout aller réussi est suivi du retour depuis (q', −p') (même routine `trajectoire`) : il doit réussir et revenir à (q, −p) à 1e-7.

## 3. Gates (statut calculé par `reverdict_newton.py`)
| ID | énoncé | PASS ssi |
|---|---|---|
| NEWTON-PROTOCOLE | le fichier de résultats est une exécution de verdict du protocole gelé, types stricts (booléens refusés) : schéma, `hors_verdict` faux, 8 cellules exactes (identifiant, M2, dt), graines g+1…g+4 avec g = 26110160000 + 10·cellule, tol_newton 1e-13, tol_rev 1e-9, n = 2 000 000, invariance 200 000 (compte cohérent), profil : grille exacte x_k = −π + (k + ½)·2π/400, 400 nœuds, 20 000 essais, comptes entiers dans [0, 20 000], résumés complets | aucun écart ; toute surcharge d'environnement marque le fichier hors verdict |
| NEWTON-ABSENCE-DE-BIAIS | E_rej contre la vérité, 8 cellules × 2 observables = 16 comparaisons | toutes valides (domaine) et max|t| < 3,88 |
| NEWTON-PUISSANCE | E_drop et E_retry contre la vérité, cellules à taux d'échec ≥ 4 % | au moins une comparaison valide avec |t| ≥ 3,88 (ROUGE exigé) ; SKIP si aucune comparaison valide |
| NEWTON-PREDICTION | biais mesuré de E_drop et E_retry contre le biais prédit (cellules ≥ 4 %), erreur du profil propagée avec la dérivée complète db/df_k = −w_k[(O_k − E O) − b]/(1 − f̄) | toutes les comparaisons attendues (2 mutants × 2 observables × cellules ≥ 4 %) valides ET max|t_pred| < 3,88 ; SKIP si l'une est invalide |
| NEWTON-INVARIANCE | allers réussis suivis d'un retour réussi et revenant au départ | zéro exception, ≥ 10⁴ allers par cellule |
| NEWTON-SU2-REJEU | rejeu de la reprise n2bis (L = 6, β = 2,8, dt = 0,01, 2 pas, 1000 trajectoires, graines 26102612821 / 26102612921) ; fichier OBLIGATOIRE | 29 échecs et 939 acceptées retrouvés (recalculé par le script depuis les comptes) |
Statut : BIAIS_DETECTE si NEWTON-ABSENCE-DE-BIAIS est FAIL ; COUT_SANS_BIAIS si tous les gates sont PASS ; HORS_PROTOCOLE si NEWTON-PROTOCOLE échoue ; INDETERMINE sinon (SKIP puissance, rejeu absent, etc.). Les résultats sont rapportés par mutant et par cellule (analyse.json) : la puissance affirmée est celle effectivement observée.
Rapportés sans seuil : taux d'échec, acceptation, τ_int, ESS par trajectoire (coût) ; rejeu à max_newton = 500 des échecs SU(2) ; fenêtres de 100 trajectoires.

## 4. Portée et limites déclarées
Conclusion autorisée : « coût sans biais détecté sur CT-7 périodique, pour les échecs de Newton et de réversibilité couverts ». 
Le jouet (1 contrainte) ne prouve rien sur SU(2) : le rejeu SU(2) ne vérifie que la reproduction des échecs et leur résolution à plafond 500. Les échecs d'évaluation (branche du logarithme) ne sont pas couverts : ils changent le domaine, pas la convergence. Le seuil gelé « ≤ 1 % » de FIB-CONTR n'est pas modifié : son SKIP reste le résultat formel ; ce re-verdict dit seulement ce qu'il vaut sur le fond. L'invariance est validée numériquement à la précision testée (tolérance de réversibilité 1e-9 sur la position, 1e-7 sur la position et l'impulsion au retour), sans borne rigoureuse du biais stationnaire.
