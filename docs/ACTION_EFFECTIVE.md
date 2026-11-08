# `kvu rg action` — couplages d'une action effective tronquée

## Objet
Le module `rg action` mesure les couplages d'une action effective **tronquée** sur des configurations
de jauge SU(2) déjà bloquées (ou non bloquées). L'action d'essai ne retient que quatre types de boucles
fondamentales : la plaquette **P** (1×1), le rectangle **R** (1×2), la chaise **C** et la torsade **T**.
Les couplages sont estimés par deux équations d'estimation exactes en espérance sous l'action
d'essai, sans aucun Monte-Carlo inverse. Leurs solutions à taille finie peuvent être biaisées.
Un échantillonneur de Gibbs de l'action tronquée sert d'instrument de contrôle.

## Conventions
- Réseau périodique L⁴ ; lien U_μ(x) de x vers x + μ̂ ; quaternions q = (a₀, a₁, a₂, a₃), U = a₀ + i(a₁σ₁ + a₂σ₂ + a₃σ₃).
- Indice de lien li = μ·V + s avec s = ((x₀L + x₁)L + x₂)L + x₃, V = L⁴.
- Chemin : pas (d, σ), d ∈ {0,1,2,3}, σ = ±1. Pas (d,+1) : multiplier à droite par U_d(pos) puis pos ← pos + d̂ ;
  pas (d,−1) : pos ← pos − d̂ puis multiplier à droite par U_d(pos)†.

## Énumération des boucles
Les quatre types sont énumérés par force brute (longueurs 4 et 6) pour chaque direction μ, en imposant :
premier pas = (μ,+1), fermeture **exacte dans ℤ⁴** (déplacement total nul, pas seulement retour au site sur le tore),
et absence de rebroussement cyclique. La classification est :
- longueur 4 ⇒ **P** ;
- longueur 6 avec 2 directions distinctes ⇒ **R** ;
- longueur 6 avec 3 directions distinctes ⇒ **C** si les sites visités contiennent les quatre coins d'un carré unité, sinon **T**.

Comptes par lien (n_lien) : P 6, R 18, C 72, T 24 ; par site : 6, 12, 48, 16.
Longueurs |γ| : 4, 6, 6, 6. Poids R4 = n_lien × (|γ|−1) : 18, 90, 360, 120.

## Estimateurs
### Pseudo-vraisemblance (PL)
La loi conditionnelle d'un lien est de von Mises–Fisher : densité ∝ exp(Re(U_ℓ A_ℓ(β))),
A_ℓ(β) = Σ_γ β_γ A_ℓ^γ (agrafes). Le maximum de la pseudo-vraisemblance est concave ; il est résolu
par Newton protégé (recherche linéaire, Cholesky). Les couplages PL restent signés. Le départ
et les candidats sont contrôlés avant les évaluations de Bessel : |β|∞ > 10⁴ est refusé comme « maximum à l'infini » ; un candidat excessif ou non fini est rejeté.
Pour F = {P}, l'équation PL coïncide avec celle de `rg::flot::sd::beta_sd` (mode Exact)
pour une racine positive dans le domaine de recherche de ce témoin, **[0, 50]**.
Une racine PL négative ou supérieure à 50 n'est pas couverte par cette équivalence au témoin.
Le JSON indique le domaine et le résidu du témoin de Wilson ; sa valeur est `null` si aucune
racine admissible n'existe ou si le résidu dépasse 10⁻¹⁰·max(1, |Σw|). Une erreur jackknife
indisponible est également `null`. Le témoin ne borne ni ne contraint les couplages PL.

### Schwinger–Dyson linéaire (SD)
L'intégration par parties sur le groupe donne le système linéaire M β = b, où
b_γ = 3 Σ_ℓ Re(U_ℓ A_ℓ^γ) et M_{γδ} = Σ_ℓ Im(U_ℓ A_ℓ^γ)·Im(U_ℓ A_ℓ^δ). C'est un estimateur témoin,
dont l'équation est exacte en espérance si l'action vraie appartient à la famille.
La matrice est normalisée avant Cholesky et le calcul spectral ; le refus
λ_min ≤ 10⁻¹² λ_max est indépendant de son échelle. Jacobi contrôle sa convergence.

L'écart PL−SD (`ecart_pl_sd`) est un diagnostic statistique **possible** de troncature.
Un accord ne caractérise pas l'appartenance de l'action vraie à la famille. Cet écart n'est
ni une erreur-type de l'action ni une borne du reste omis ; les fluctuations à taille finie
et les conditions d'échantillonnage interviennent aussi.

## Jackknife
Les groupes sont les chaînes (défaut) ou des blocs de configurations consécutives (mode `blocs:B`,
tailles égales à une unité près). La valeur centrale est calculée sur toutes les données ; chaque
réplicat omet un groupe. Erreur : σ² = (G−1)/G Σ_g (θ_g − θ̄)². Les quantités dérivées (N4, critère P+R,
différences PL−SD) sont recalculées sur chaque réplicat.
La différence PL−SD utilise les réplicats **appariés** : d_g = β_PL,g − β_SD,g, puis
la formule jackknife appliquée aux d_g ; les erreurs PL et SD ne sont pas additionnées
en quadrature. Le calcul préserve ainsi leur covariance.

Cette formule suppose des groupes de tailles comparables : le module refuse un rapport
taille maximale / taille minimale > 2. Les chaînes doivent être stationnaires et indépendantes ;
les blocs doivent être longs devant les temps d'autocorrélation. Ces conditions physiques
restent à vérifier sur les ensembles : un simple découpage ne les établit pas.

## Norme R4 et critère P+R
La norme R4 est N4 = Σ_{γ ∈ F} n_lien(γ) (|γ|−1) |β_γ| = 18|β_P| + 90|β_R| + 360|β_C| + 120|β_T|.
Le critère « trou de type Dobrushin » est N4 < 4 (α₄ = N4/4). Le critère P+R est
c_PR = |β_P| + 5|β_R|, seuil 2/9 ; il ignore C et T. **Ces critères sont des mesures de schéma,
jamais une preuve** de confinement.
N4 et c_PR ne sont pas différentiables lorsqu'un couplage s'annule : leurs erreurs jackknife
ne justifient pas automatiquement des intervalles symétriques gaussiens près de zéro.

## Échantillonneur de Gibbs
`gibbs::Gibbs` simule l'action tronquée S_β (bain thermique séquentiel exact + sur-relaxation).
C'est un **instrument de contrôle** (fermeture, comparaison avec le bain thermique de Wilson),
pas un générateur de production. Aucun parallélisme : sorties déterministes.
Pour k < 10⁻³, M2 tire directement depuis Haar et accepte avec probabilité exp(Re(UA)−k).
Ce rejet exact évite les annulations du noyau historique aux très petits k > 0, sans modifier
`heritage/`. Les plafonds d'essais restent comptés. Sous Haar, Var(½Tr U)=1/4 ; pour
50 champs L=4, l'erreur de la plaquette moyenne est 1/(2√(50·6·4⁴)) ≈ 0,001804.

## Interface en ligne de commande
```
kvu rg action --ensemble DIR [--filtre MOTIF] [--operateur …] [--pas K] [--famille P,R,C,T]
              [--estimateurs pl,sd] [--jackknife chaines|blocs:B] [--max-configs N]
              [--sortie DIR] [--fils N] [--memoire-max-go X]
```
- `--ensemble` : une chaîne (dossier avec `manifest.json` et `cfg_*.npy` / `cfg_*.npy.zst`) ou un parent de chaînes.
- `--pas K` : K pas de blocage par l'opérateur `--operateur` (défaut `3centre`, utilisé seulement si K ≥ 1).
- Sortie `<sortie>/action.json` (refus d'écraser, écriture atomique). Code de sortie 0 si tous les
  estimateurs sont « ok », 8 si un estimateur est en échec (le JSON est écrit quand même).
- La garde mémoire PL est décidée avant toute allocation de G. Les statistiques sont accumulées
  directement dans les groupes, sans copie des G par configuration. Avec `--estimateurs sd`,
  ou si PL est refusé pour mémoire, aucun G n'est conservé ; SD et le JSON restent produits.
- L'opérateur est validé avant la découverte des chaînes ; 3^K est calculé sans débordement.
  Chaque champ doit avoir le côté annoncé par le manifeste, puis le côté attendu après chaque
  blocage. Les incohérences sont des erreurs d'usage (code 2).

## Schéma du JSON (extrait)
```json
{
  "format": "kvu-rg-action/1",
  "famille": ["P","R","C","T"],
  "pl": {
    "etat": "ok",
    "couplages": {"P": {"valeur": ..., "erreur": ..., "replicats": [...]}, ...},
    "r4": {"norme": {"valeur": ..., "erreur": ...}, "alpha4": ..., "inferieur_a_4": ...},
    "critere_pr": {"valeur": {...}, "seuil": 0.2222222222222222, "inferieur_au_seuil": ...},
    "correlations": [[...], [...], [...], [...]],
    "iterations": ..., "norme_gradient": ..., "valeurs_propres_information": [...]
  },
  "sd": { /* mêmes clés, avec "valeurs_propres_m" */ },
  "beta_sd_un_couplage": {"valeur": ..., "erreur": ..., "residu": ..., "domaine": [0, 50], "note": "..."},
  "ecart_pl_sd": {...},
  "note": "mesure de schéma : couplages d'une action tronquée par identités exactes ; aucune lecture physique"
}
```
`correlations` a n lignes de n nombres, dans l'ordre de `famille`. Un estimateur non demandé
est absent. `ecart_pl_sd` n'existe que si les deux estimateurs sont demandés et réussissent.
Pour le témoin de Wilson sans racine admissible (notamment champ froid ou racine signée
négative), `valeur`, `erreur` et `residu` sont `null`.

## Tests et gates
- Tests d'intégration `tests/rg_action.rs` : champ froid, Haar, non-régression SD, Wilson non bloqué à β connu,
  fermeture et troncature par Gibbs, comparaison Gibbs / bain thermique, invariance de jauge,
  déterminisme CLI, pipeline bloqué, refus CLI.
- Gates `rg action-gates` : `RG-ACTION-ENUM`, `RG-ACTION-SD1`, `RG-ACTION-FERME`,
  `RG-ACTION-WILSON-b<β>` (β = 2,3 ; 2,5 ; 2,8 ; 4), `RG-ACTION-MUT` (mutants de contrôle).
- `action-gates --json` sérialise les lignes et le verdict global. `--seul` refuse les
  identifiants inconnus et les sélections vides (code 2). Code 0 si tout est vert, 1 sinon.
  Les données Wilson absentes donnent SKIP ; un manifeste ou un champ invalide donne FAIL.
  Le mutant I₁/I₀ possède sa propre dérivée s′=1−s/k−s², avec s/k→1/2 et
  (s′−s/k)/k²→−1/8. La comparaison à 4σ exige sa convergence centrale et celle des réplicats ;
  un échec numérique donne FAIL, jamais un contrôle négatif réussi.
- T6 impose σ_R < 0,025, σ_C < 0,0125 et σ_T < 0,015 pour PL **et** SD.
  T9 lit douze configurations, T10 six ; les indices commencent à 1 (`cfg_000000` exclu).
  T12 construit un ensemble indépendant pour chaque refus et vérifie le diagnostic du module.
- Les seuils exprimés en « σ » gardent un risque d'échec statistique non nul. Avec peu de
  groupes, une loi de Student à peu de degrés de liberté est plus pertinente qu'une lecture
  gaussienne ; plusieurs comparaisons augmentent le risque global. Vérifier la stabilité des
  erreurs quand la longueur des blocs augmente, sans relâcher les seuils d'acceptation.
- Arbitrage du chef de projet : cible de durée de `rg_action` **< 6 min en release sur 8 cœurs**,
  remplaçant < 90 s. Les deux ensembles T5 (L=8, 192 configurations chacun) sont générés dans
  deux fils avec les mêmes graines et le même ordre local des opérations qu'en séquentiel.
  T8 rapporte l'écart de jauge à max(1, max_γ |β_γ|), au seuil 10⁻¹⁰.

## Limites
- PL et SD sont deux **projections** différentes ; leur écart peut signaler une troncature,
  sans réciproque ni borne quantitative du reste.
- Terme adjoint et boucles de longueur > 6 sont hors périmètre.
- GPU non couvert (tout le module est CPU, FP64).
- Mémoire de PL ∝ nombre de liens × n(n+1)/2 (garde `--memoire-max-go`, défaut 4 Gio).
