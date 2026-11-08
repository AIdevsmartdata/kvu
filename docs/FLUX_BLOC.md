# Flux de bloc 3×3 : `kvu rg flux`

## Objet
Le module `rg flux` mesure, pour les configurations SU(2) 4D, la fraction des
faces de bloc 3×3 dont l'holonomie dépasse des seuils prédéfinis, ainsi que le
**flux cohérent** d'une face, qui distingue un flux aligné (fond dangereux)
d'un angle obtenu par accident (bruit UV). Il porte en Rust le script
`boucles.py` du chantier `desert_debut_2026-11` et ajoute le flux cohérent
défini par la référence indépendante `reference_flux_py.py`.

## Définitions

### Faces n×n et angle propre
- Plans : (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
- Ligne : S_μ⁽ⁿ⁾(x) = U_μ(x) · U_μ(x+μ̂) · … · U_μ(x+(n−1)μ̂), produit de gauche
  à droite ; S⁽⁰⁾ = identité.
- Face de coin x, plan (μ,ν), taille n :
  W = ((S_μ⁽ⁿ⁾(x) · S_ν⁽ⁿ⁾(x+nμ̂)) · S_μ⁽ⁿ⁾(x+nν̂)†) · S_ν⁽ⁿ⁾(x)†.
  L'association des produits est EXACTEMENT celle de `boucles.py` ; elle est
  imposée bit à bit car l'associativité mathématique ne garantit pas les mêmes
  arrondis.
- x = ½ Tr W ; xc = min(1, max(−1, x)).
- Angle propre (convention programme) : Θ = 2 arccos(xc) ∈ [0, 2π].
  **Repliement** : un fond abélien uniforme F_{μν} = Φ donne Θ = 9Φ tant que
  9Φ ≤ 2π, puis Θ = 4π − 9Φ. La trace ne distingue pas 2π(1+δ) de 2π(1−δ).
- Contrôle de précision (pas pour les statistiques) :
  Θ_atan = 2·atan2(|W⃗|, W[0]), bien conditionné près de 0 et de 2π.

### Statistiques brutes
Seuils sur xc (issus des PREREG de `desert_debut_2026-11`, §1) :

| Nom      | Condition        | Seuil xc            | Θ correspondant |
|----------|------------------|---------------------|-----------------|
| T1_pi    | xc < 0           | 0.0                 | Θ > π           |
| T2_3pi2  | xc < −√½         | −0.7071067811865476 | Θ > 3π/2        |
| C32      | xc ≤ −cos(π/32)  | −0.9951847266721969 | Θ ≥ 2π(1−1/32)  |
| C16      | xc ≤ −cos(π/16)  | −0.9807852804032304 | Θ ≥ 2π(1−1/16)  |
| C8       | xc ≤ −cos(π/8)   | −0.9238795325112867 | Θ ≥ 2π(1−1/8)   |

Comptes exacts, fractions = comptes/N. Sommes sx, sx2, sΘ², sΘ⁴, xmin.
Histogramme facultatif de α = arccos(xc) sur 3600 classes.

### Flux cohérent
Pour chaque face n×n de coin x, plan (μ,ν), on transporte chaque plaquette
interne au coin :
- plaquette en y = x + aμ̂ + bν̂ :
  U_p(y) = (U_μ(y)·U_ν(y+μ̂))·(U_μ(y+ν̂)†·U_ν(y)†).
- transport T_ab = S_μ⁽ᵃ⁾(x) · S_ν⁽ᵇ⁾(x+aμ̂).
- Ũ = (T_ab · U_p(y)) · T_ab†.
- vecteur de flux : v = partie vectorielle de Ũ, Θ_p = 2 atan2(|v|, Ũ[0]).
  Si v ≠ 0, m = max_i |v_i|, u = v/m, puis φ_ab = Θ_p·(u/|u|).
  Cette normalisation reste robuste jusqu'aux plus petits sous-normaux et
  évite le débordement près de −I. Si v = 0, φ_ab = 0.
- flux cohérent Φ_coh = |Σ_ab φ_ab| (NON replié : peut dépasser 2π).
- somme des angles S = Σ_ab Θ_p ; cohérence C = Φ_coh/S (C=1 si S=0).
- plaquette ambiguë si Θ_p > 2π − 2·10⁻⁶ (compte `n_ambigues`).

**Propriétés** :
- **Invariance de jauge** : pour un choix FIXÉ des chemins de transport (ici
  d'abord μ puis ν), tous les Ũ d'une face tournent par le même g(x) ; Φ_coh
  est donc une observable invariante de jauge. La dépendance aux chemins
  n'empêche pas l'invariance : chaque choix fixé définit une observable
  invariante de jauge différente. C'est la convention retenue ici, identique à
  `reference_flux_py.py` et à la spécification M3.
- Fond abélien uniforme ⇒ Φ_coh = n²Φ exactement, C = 1.
- Champ aléatoire ⇒ C ≈ 1/n.

Seuils sur Φ_coh : Q1_2 (> π), Q3_4 (> 1.5π), Q31_32 (≥ 2π·31/32), Q1 (≥ 2π).
Comptes conjoints de cohérence C ≥ 0.5, 0.8, 0.95.

### Pavage
Option `--pavage tous|0|1|2` : pour o ∈ {0,1,2} et L multiple de 3, seules les
faces dont les quatre coordonnées du coin valent o modulo 3 sont retenues
(6·(L/3)⁴ faces). o=0 correspond aux coins des blocs (`dec`/`ape`), o=1 aux
centres des blocs (`3centre`).

### Plaquette grossière
Deux modes :
- calcul : K pas (défaut 1) de l'opérateur `--grossier-operateur` (défaut
  `3centre`) via `rg::lissage::bloquer_operateur`, puis statistiques n=1 du
  réseau grossier. L doit être divisible par 3^K.
- fichier : `--grossier-fichier Y.npy`, statistiques n=1 du fichier.

## CLI

```text
kvu rg flux --entree X.npy [--n 1,2,3] [--coherent 3] [--pavage tous|0|1|2]
            [--grossier] [--grossier-operateur SPEC] [--grossier-pas K]
            [--hors-famille] [--grossier-fichier Y.npy] [--histogramme]
            [--fils N] [--sortie R.json]

kvu rg flux --liste LISTE.jsonl --sortie LIGNES.jsonl [--coherent 3] [--fils N]

kvu rg flux-gates [--desert DIR] [--max-configs N] [--seul ID,…]
                  [--sans-mutants] [--json] [--fils N]
```

Les sorties JSON respectent le schéma `kvu-rg-flux/1` : `format`, `version`,
`entree`, `sha256_entree`, `L`, `pavage`, `brut`, `coherent`, `grossier`
(optionnel), `seuils`, `note`. Aucune durée ni nombre de fils : le résultat est
identique à l'octet quel que soit `--fils`.

Les réductions utilisent, **pour chaque plan séparément**, les tranches de
sites physiques [0,4096), [4096,8192), etc., avec une dernière tranche
éventuellement incomplète. Le pavage filtre les coins à l'intérieur de ces
tranches ; une tranche vide ne contribue pas à `xmin`. Les sommes partielles
sont fusionnées séquentiellement dans l'ordre (plan, premier site).
L'écriture réserve `<sortie>.<pid>.part` avec `create_new`, synchronise le
contenu, puis publie par lien physique sans remplacement. Seul un temporaire
créé par l'appel est nettoyé ; une collision conserve les fichiers existants.
La ligne INFO inclut aussi `grossier_n1_f_T1` si le grossier est demandé.

Le mode liste porte `traiter.py` : chaque ligne d'entrée décrit un travail
(`kind` fin|gros1|gros2, `path`, clés `cell`, `chain`, `idx`, `L`, `beta`,
`src` recopiées). Une ligne de sortie par entrée, sans histogramme.
Un `L` présent doit être un entier positif conforme au fichier ; cette
validation est commune au mode liste et à G-REPRO. Un `--max-configs` non entier
est une erreur de configuration (rc 2).

## Gates
- `RG-FLUX-ABELIEN` : huit cas de fonds abéliens (§5 de la spec), valeurs par
  face, comptes exacts, grossier `3centre`.
- `RG-FLUX-FROID` : champ froid L=6, x=1 sur chaque face, comptes nuls.
- `RG-FLUX-JAUGE` : champ de Haar + jauge de Haar : comptes identiques, écarts
  ≤ 1e-13 (x) et ≤ 1e-12 (Φ_coh).
- `RG-FLUX-REF-PY` : comparaison aux trois fixtures de `reference_py.json`.
  Tolérances : comptes identiques ; sommes relatives ≤ 1e-12 (sx, sx2, sΘ²,
  sΘ⁴) ; **xmin : 1e-13 en absolu** (voir la note ci-dessous) ; cohérent :
  comptes identiques, sommes relatives ≤ 1e-11, phi_max ≤ 1e-12 ;
  histogrammes : classes non vides IDENTIQUES dans les deux sens.
- `RG-FLUX-GROSSIER` : blocage `3centre` de `cfg_s7_q64.npy`, égal à
  `ubar_s7_q64.npy`.
- `G-REPRO` : si `--desert` fourni, recalcule le mode liste et compare aux
  lignes de référence de `desert_debut_2026-11`. Chaque ligne de référence
  attendue (fin et gros1) doit exister. Ensembles d'observables stricts :
  n1/n2/n3 pour fin, n1 pour gros1. Les sept champs Python obligatoires sont
  `N`, `comptes`, `sx`, `sx2`, `sTheta2`, `sTheta4`, `xmin` ; `fractions` et
  `n_ambigues` sont ignorées, l'histogramme est comparé seulement si la référence
  le fournit. Comptes identiques, scalaires finis, sommes relatives ≤ 1e-10,
  xmin absolu ≤ 1e-13. Zéro ligne comparée ⇒ FAIL. Le premier écart indique
  la ligne, la clé et les valeurs calculée/de référence (ou leur absence).
  SKIP si `--desert` absent ou fichiers manquants ; un SKIP donne rc 1.
- `RG-FLUX-MUT` : mutants `SansTransport` et `PlanManquant` doivent faire
  échouer respectivement la jauge et les comptes de la référence Python,
  y compris après neutralisation de la différence de N.

## Note sur la tolérance de xmin
La tolérance de comparaison de `xmin` est **1e-13 en absolu** (arbitrage du
chef de projet, validé en relecture). L'écart résiduel observé sur la fixture
c16 vient de la conversion c16 → quaternion : `kvu` moyenne les entrées
symétriques de la matrice (`mat_to_quat`), tandis que `boucles.py` lit la
première ligne ; les écarts sont de l'ordre de 1e-16 par lien, amplifiés par
les 12 produits d'une face n=3. Les comptes restent exigés IDENTIQUES.

## Limites
- Le flux cohérent d'une face brute reste dominé par l'UV aux couplages du
  désert ; **une fraction de faces au-dessus d'un seuil majore la fraction de
  fonds dangereux au sens de la SYNTHÈSE (condition nécessaire, non
  suffisante)**.
- Φ_coh dépend du choix des chemins de transport ; chaque choix fixé définit
  une observable invariante de jauge. La comparaison aux références Python
  suppose la même convention (d'abord μ puis ν).
- Les angles propres Θ sont repliés par la trace ; seul Φ_coh détecte les flux
  > 2π (cas m=19).
