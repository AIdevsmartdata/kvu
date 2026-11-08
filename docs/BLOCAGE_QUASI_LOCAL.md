# Blocage de facteur 3 par opérateurs quasi-locaux (WP22) : conventions, définitions, famille, supports

Rédigé par le responsable du lot WP22 (Opus, tranche 4 du programme `kvu`), le 27/10/2026, **avant toute ligne de code** du lot.
Ce document fixe ce que font les options `--operateur` de `kvu rg bloc` et de `kvu rg flot`. Il ne contient **aucun résultat
physique** : la lecture des sorties (localité de l'action effective, désert) appartient au chantier vision (BILAN_VISION §38, §40).
Les calculs qui portent les affirmations chiffrées sont dans `crates/kvu/validation/rg_ql/derivations_ql.py` (SymPy et numpy,
D1-D5, RC 0) et `crates/kvu/validation/rg_ql/support_py.py` (parcours en largeur indépendant de kvu).

## 1. Objet

Un pas de blocage de facteur L_b = 3 transforme une configuration SU(2) de côté L (L ≡ 0 mod 3, L ≥ 3) en une configuration de côté
L/3, au même format de fichier (`c16` ou `q64`). Opérateurs disponibles :

| spécification | nom | idée | origine |
|---|---|---|---|
| `3centre` (défaut) | moyenne non linéaire de Bałaban | log/exp de 10 473 holonomies par lien grossier | ARCHITECTURE §4.1, WP6 (inchangé au bit) |
| `dec` | décimation pure | produit des 3 liens alignés | BLOC2 `operateurs.py:39-53` |
| `ape:<α>:<n>` | lissage APE puis décimation | n pas APE simultanés sur le réseau fin, puis `dec` | BLOC2 `operateurs.py:74-102, 148-157` |
| `flot:<t>[:<ε>]` | flot de Wilson puis décimation | flot RK3 de Lüscher jusqu'au temps t (pas ε, défaut 0,01), puis `dec` | `obs::ref_cpu` (OBS-3), Lüscher 1006.4518 |
| `agr:<c>` (facultatif) | « droit + agrafes » projeté | lien de longueur 3 + c × 6 agrafes de longueur 3, projeté | BLOC2 `operateurs.py:56-71` |

BLOC2 = `chantiers/vision_20261025/calculs/BLOC2_operateurs/code/bloc2/operateurs.py` (lecture seule, sha256 au manifeste des
références), écrit et testé par le chantier vision le 26/10 (G-JAUGE, G-DEC-EXACT, VERDICT.md).

## 2. Conventions (celles de kvu, inchangées)

- **Liens** : U(x, μ) de x vers x + μ̂ ; quaternion q = (a₀, a₁, a₂, a₃), U = a₀ + i(a₁σ₁ + a₂σ₂ + a₃σ₃), matrice
  [[a₀ + i a₃, a₂ + i a₁], [−a₂ + i a₁, a₀ − i a₃]] (`heritage/su2_hb.rs:97-99`, `docs/CONVENTIONS.md` §1) ; produit `qmul`
  (`su2_hb.rs:68-75`, même formule que BLOC2 `qmul` à l'association près de la composante 0) ; U† = conjugué (`qdag`).
- **Rangement** : lien li = μ·V + s, s = ((x₀L + x₁)L + x₂)L + x₃, V = L⁴ (`obs/ref_cpu.rs:13`) ; c'est aussi l'ordre (4, L, L, L, L, 4)
  des tableaux numpy de BLOC2. Tore périodique.
- **Lien grossier des opérateurs à décimation** (`dec`, `ape`, `flot`, `agr`) : V(X, μ) va du site fin **3X** au site fin 3X + 3μ̂
  (coin du bloc B(X) = 3X + {0,1,2}⁴). **Attention** : le lien grossier de `3centre` va du CENTRE c(X) = 3X + (1,1,1,1) à c(X + μ̂).
  Les deux sorties ont le même format (L/3)⁴, mais leurs liens ne sont pas posés aux mêmes points du réseau fin (translation
  (1,1,1,1)). Toute comparaison qui dépend de la position (distance, réponse) doit utiliser le point d'attache de l'opérateur.
- **Covariance exigée** : si U(x, μ) → g(x) U(x, μ) g(x + μ̂)†, alors V(X, μ) → g(3X) V(X, μ) g(3X + 3μ̂)† (`dec`, `ape`, `flot`,
  `agr`) ; pour `3centre`, V → g(c) V g(c′)† (RG-JAUGE de WP6).
- **Projection SU(2)** : P(s) = s/|s|, |s| = norme euclidienne des 4 composantes. Pour SU(2), ½ Re tr(W† S) = w·s (produit scalaire des
  quaternions) pour tout S somme réelle d'éléments de SU(2) ; P(s) est donc l'élément de SU(2) qui maximise ½ Re tr(W† S). **Refus**
  (erreur, rc ≠ 0) si |s| < 1e-12 : la projection est ambiguë (même règle que BLOC2 `operateurs.py:31-36`).
- **Déterminisme** : chaque lien (fin lissé ou grossier) est calculé indépendamment à partir du champ de l'étape précédente, sans
  réduction entre liens ; la sortie et le manifeste sont **identiques à l'octet quel que soit `--fils`** (comme RG-FILS de WP6).
- **Précision** : F64 partout (CPU). Le GPU n'est pas couvert par ce document (facultatif, voir §8).

## 3. Définitions exactes (ordre des opérations compris)

### 3.1 `dec`
V(X, μ) = (U(x, μ) · U(x + μ̂, μ)) · U(x + 2μ̂, μ), x = 3X (produit évalué de gauche à droite, comme BLOC2 `droit`).
Identité : le produit ordonné des liens grossiers le long d'une boucle grossière R × T est le produit ordonné des liens fins le long
de la boucle fine 3R × 3T (mêmes liens, même ordre) ; les deux ne diffèrent que par l'association des produits (arrondi).

### 3.2 `ape:<α>:<n>` (α ∈ ]0, 1[, n entier ≥ 0)
Un pas APE, **mise à jour simultanée** (toutes les directions et tous les sites lisent le champ du pas précédent) :

  s(x, μ) = (1 − α) U(x, μ) + (α/6) · Σ_{ν ≠ μ, ν = 0..3 croissant} [ a_ν(x, μ) + b_ν(x, μ) ],
  a_ν = U(x, ν) U(x + ν̂, μ) U(x + μ̂, ν)†,  b_ν = U(x − ν̂, ν)† U(x − ν̂, μ) U(x − ν̂ + μ̂, ν),
  U′(x, μ) = P(s(x, μ)).

Ordre exact (celui de BLOC2 `agrafes`/`ape_pas`) : tot = 0 ; pour ν croissant ≠ μ : tot ← tot + (a_ν + b_ν) ; s = (1 − α)·U + (α/6)·tot ;
chaque produit à trois facteurs est évalué (A·B)·C. Après n pas, `dec` sur le champ lissé. **n = 0 ⇒ `dec` exactement (au bit).**

Forme équivalente de la littérature : [lecture] Greensite, *The confinement problem in lattice gauge theory*, hep-lat/0301023,
éq. (55), p. 23 — `rageq` méthode arxiv_tex, confiance du numéro haute — `reading_list/equations/arxiv_src/hep-lat_0301023/src/revp.tex:1462-1470`
— vérifié sur l'image : non. Elle s'écrit U′ = U + c Σ_{k≠i}[U_k(x) U_i(x+k̂) U_k†(x+î) + U_k†(x−k̂) U_i(x−k̂) U_k(x−k̂+î)], puis
projection sur le groupe (« project the "thick" link variables U′_i back to the SU(N) group manifold », même page), pour des liens
spatiaux ; Greensite l'attribue à la réf. [APE] de sa bibliographie (Albanese et al.), que nous n'avons pas ouverte : nous ne la citons
pas. Comme P est positivement homogène (P(λs) = P(s), λ > 0), notre pas est celui de (55) en 4D avec **c = α/(6(1 − α))** (α = 0,5 ⇒
c = 1/6), au détail près de l'arrondi (nous suivons l'écriture de BLOC2, pas celle de (55)).

### 3.3 `flot:<t>[:<ε>]` (t ≥ 0, ε > 0, défaut ε = 0,01, N = t/ε entier)
Flot de Wilson V̇_t(x, μ) = −g₀² {∂_{x,μ} S_w(V_t)} V_t(x, μ), V_0 = U : [lecture] Lüscher, *Properties and uses of the Wilson flow in
lattice QCD*, arXiv:1006.4518, éq. (1.4), p. 2 — source LaTeX `sect1` (archive arXiv, sha256 a5aaeb37…) et image de la page 2 — vérifiée : oui.
Intégration par le schéma de Runge-Kutta de l'annexe C, éq. (C.2)-(C.3), p. 20 :
W₀ = V_t ; W₁ = exp{¼ Z₀} W₀ ; W₂ = exp{(8/9) Z₁ − (17/36) Z₀} W₁ ; V_{t+ε} = exp{(3/4) Z₂ − (8/9) Z₁ + (17/36) Z₀} W₂ ; Z_i = ε Z(W_i)
— source LaTeX `appc` et image de la page 20 du PDF (sha256 63426411…) vérifiées : oui.

Implémentation : **celle de kvu, sans réécriture** : `obs::ref_cpu::flot_n_pas(champ, ε, N)` (`obs/ref_cpu.rs:344-353`), qui applique N
fois `pas_rk3` (`ref_cpu.rs:321-328`) avec Z(x, μ) = −Ta(U(x, μ) · A(x, μ)), A = somme des 6 agrafes FERMANTES (`su2_hb::staple`,
U·A = somme des plaquettes qui commencent par le lien), exponentielle exacte de su(2) puis renormalisation du quaternion
(`ref_cpu.rs:300-319`). Ce flot est celui de la gate OBS-3 (verte, T7). Puis `dec`.
N = t/ε doit être entier : on prend N = arrondi(t/ε) et on **refuse** si |N ε − t| > 1e-9 · max(1, t). **t = 0 ⇒ N = 0 ⇒ `dec` au bit.**

Normalisation du temps (vérifiée par calcul, D4, et par la gate RG-QL-NORM) : au premier ordre abélien, ce Z multiplie une onde plane
transverse θ_μ(x) = a cos(p x_ρ) (ρ ≠ μ) par e^{−t p̂²}, p̂² = 4 sin²(p/2), c'est-à-dire le noyau de chaleur de Lüscher éq. (2.12),
K_t(z) = e^{−z²/4t}/(4πt)^{D/2} (même source, p. 4 ; image vérifiée : oui), dans sa version de réseau. Le temps t de kvu est donc celui de
Lüscher.

### 3.4 `agr:<c>` (c ≥ 0 ; facultatif)
S(x) = (U(x, μ) U(x+μ̂, μ)) U(x+2μ̂, μ) ; tot = S(x) ; pour ν croissant ≠ μ :
tot ← tot + c·(A₊ + A₋), A₊ = (U(x, ν) S(x+ν̂)) U(x+3μ̂, ν)†, A₋ = (U(x−ν̂, ν)† S(x−ν̂)) U(x−ν̂+3μ̂, ν) ; V(X, μ) = P(tot), x = 3X.
(BLOC2 `epais_agr`.) c = 0 donne P(S), égal à `dec` à l'arrondi près seulement (pas au bit).

### 3.5 `3centre`
Inchangé (ARCHITECTURE §4.1, `rg/bloc.rs`) : sortie, manifeste (`kvu-rg-bloc/1`), lignes de `rg gates` et de `rg flot-gates`
identiques à l'octet à celles de `main`/`wp9`.

## 4. La famille « rayon de lissage ⩽ bloc » : règle exacte appliquée, et pourquoi

Le rayon contrôlé est un **rayon quadratique moyen de lissage**, pas un rayon de support (le support exact est au §5 : il est bien
plus grand). Règle :

| opérateur | rayon r | dans la famille ssi | justification |
|---|---|---|---|
| `3centre`, `dec`, `agr:c` | 0 (aucun lissage préalable) | toujours | moyenne confinée aux blocs voisins |
| `flot:t:ε` | √(8t) | **8t ≤ 9** (t ≤ 9/8) **et ε ≤ 0,1** | (a) ci-dessous |
| `ape:α:n` | √(8 t_APE), t_APE = αn/6 | **4αn ≤ 27 et α ≤ 3/4** | (b) ci-dessous |

(a) **Flot** : « the gauge potential is averaged over a spherical range in space whose mean-square radius in four dimensions is equal
to √8t » (Lüscher 1006.4518, p. 4, juste après (2.12) ; image vérifiée). Le noyau de chaleur de RÉSEAU e^{tΔ} a aussi un second moment
exactement 8t en 4D (D5). Condition r ≤ L_b = 3 ⇔ 8t ≤ 9. Pas ε : au premier ordre, le schéma (C.2) appliqué à y′ = λy a pour facteur
d'amplification R(z) = 1 + z + z²/2 + z³/6 (D1), stable sur [z*, 0] avec z* = −2,5127 (D2) ; le spectre de −Δ en 4D est [0, 16], d'où
la stabilité linéaire pour ε ≤ 0,157 ; la famille exige ε ≤ 0,1 (marge). Lüscher rapporte des erreurs d'intégration « on the order of
10⁻⁶ if ε = 0.01 » (p. 20, image vérifiée) : d'où le défaut ε = 0,01.

(b) **APE** : t ≈ αn/6 est, dans OPERATEURS.md de BLOC2, une heuristique « non sourcée ». Nous la **dérivons** ici, sans la citer :
au premier ordre abélien (liens e^{iθσ₃}, θ petit), un pas APE est EXACTEMENT un pas d'Euler explicite, de pas ε = α/6, du flot
linéarisé : θ′_μ = θ_μ + (α/6) Σ_ν (∂_ν ∇_ν θ_μ − ∂_μ ∇_ν θ_ν) (D3, vérifié symboliquement sur les 4 directions ; c'est la version
de réseau de Ḃ_μ = ∂_ν G_νμ, partie linéaire de Lüscher (1.1)). Sur les modes transverses, le noyau de n pas, (1 + (α/6)Δ)ⁿ, a un second
moment 8nα/6 = 8 t_APE exactement (D5) et il est positif ssi 1 − 8α/6 ≥ 0, soit α ≤ 3/4 (sinon « rayon quadratique moyen » n'a plus
de sens de rayon). D'où r_APE = √(4αn/3) et la condition r ≤ 3 ⇔ 4αn ≤ 27. Valeurs (α = 0,5) : n = 6 → r = 2,000 ; n = 13 → 2,944 (dans
la famille) ; n = 14 → 3,055 ; n = 40 → 5,164 (hors famille : c'est le mutant APE40 de BLOC2). La comparaison est faite en f64 sous la
forme 4·α·n ≤ 27 (exacte pour α dyadique, ce qui couvre α = 0,5 et α = 0,75).

**Limite déclarée** : (a) et (b) sont des énoncés de LINÉARISATION (champ faible). Dans le désert (β ≈ 2,3-3), les champs sont forts ;
le rayon effectif d'un lissage non linéaire n'est pas mesuré ici. C'est précisément l'objet de l'instrument du livrable B (§7).

**Refus** : un opérateur hors famille est refusé (erreur, rc ≠ 0) **sauf** avec `--hors-famille`, qui est alors écrit au manifeste
(`"hors_famille": true`). Hors de tout domaine de validité (α ∉ ]0, 1[, n non entier ou < 0, t < 0, ε ≤ 0, t/ε non entier, c < 0,
valeur non finie, spécification inconnue), le refus est inconditionnel.

## 5. Support exact d'un lien grossier

**Distance d_P** : graphe dont les sommets sont les 4L⁴ liens du tore ; deux liens distincts sont voisins ssi ils appartiennent à une
même plaquette. Le lien (x, μ) a 18 voisins (L ≥ 3) : pour chaque ν ≠ μ, (x, ν), (x+ν̂, μ), (x+μ̂, ν) [plaquette (x ; μ, ν)] et
(x−ν̂, ν), (x−ν̂, μ), (x−ν̂+μ̂, ν) [plaquette (x−ν̂ ; μ, ν)]. Ce sont exactement les liens qui entrent dans les 6 agrafes du lien.
Chemin droit de V(X, μ) : C(X, μ) = {(3X + kμ̂, μ) : k = 0, 1, 2}. On note B(C, R) = {ℓ : d_P(ℓ, C) ≤ R}.

| opérateur | support exact S de V(X, μ) | pourquoi |
|---|---|---|
| `dec` | C (3 liens) | définition |
| `ape:α:n` | B(C, n) | un pas : U′(ℓ) dépend de ℓ (car 1 − α > 0) et de ses 18 voisins, et de rien d'autre ; récurrence |
| `flot:t:ε` (N pas) | B(C, 3N) | un pas RK3 évalue Z trois fois ; Z(ℓ) dépend de ℓ et de ses 18 voisins ; la mise à jour est locale au lien |
| `agr:c` | C ∪ {liens des 6 agrafes} (33 liens) | définition (§3.4) : (3X±…) voir `support_py.py` |
| `3centre` | union des liens des chemins de la table (Γ, segments, Γ′) | définition (ARCHITECTURE §4.1) |

« Exact » veut dire : modifier un lien fin HORS de S ne change AUCUN octet de V(X, μ) (le calcul de V ne le lit pas) ; modifier un lien
de S le change en général (gate RG-QL-SUPPORT : liens de la couche frontière d_P = R et de la couche d_P = R + 1). Le support est
calculé par kvu (parcours en largeur sur le tore, il tient compte des recouvrements quand la boule fait le tour) et écrit au manifeste
(nombre de liens fins pour V(0, 0)).

Tailles (calcul indépendant `support_py.py`, lien V(0, 0)) : boule B(C, R) sur le tore 12⁴ (82 944 liens) : R = 0 : 3 ; 1 : 45 ;
2 : 245 ; 3 : 793 ; 6 : 7 377 ; 7 : 12 281 ; 12 : 53 973 ; **13 : 62 129** ; 14 : 68 945 ; la boule couvre tout le tore à R = 23.
Sur 9⁴ (26 244 liens) : R = 13 : 25 852 (presque tout). **APE13 est « quasi local » au sens du rayon quadratique moyen (2,94 ⩽ 3),
mais son support exact couvre 75 % du tore 12⁴** ; un flot de N = 100 pas a un support B(C, 300) : tout le tore. La décroissance de la
dépendance à l'intérieur du support n'est PAS donnée par ce document : c'est la mesure du livrable B.

## 6. Ce qui est écrit au manifeste d'un bloc quasi-local

`<sortie>.manifeste.json`, schéma **`kvu-rg-bloc-ql/1`** (le manifeste de `3centre` reste `kvu-rg-bloc/1`, inchangé à l'octet) :
spécification canonique de l'opérateur et ses paramètres, rayon de lissage et règle de famille, `hors_famille`, définition et taille du
support (lien V(0, 0)), format de fichier, L d'entrée et de sortie, nombre de liens grossiers, sha256 de l'entrée et de la sortie,
défaut d'unitarité maximal de la sortie, plus petite norme |s| avant projection (APE, AGR ; diagnostic d'ambiguïté), version. Ni durée
ni nombre de fils (identité à l'octet).

## 7. Livrable B (instrument de localité, si A est vert)

`kvu rg reponse` : noyau de réponse des liens grossiers à une perturbation **covariante** d'un lien fin, en fonction de la distance
(test « T2 » du BILAN_VISION §40). Perturbation : U(ℓ₀) → exp(h X_a) U(ℓ₀), X_a = iσ_a (a = 1, 2, 3) au point de base de ℓ₀ ; réponse
de V : J_{ba} = d/dh [log(V_h V_0†)]_b à h = 0, matrice 3 × 3 dont la norme de Frobenius est invariante de jauge (rotation à gauche par
g(3X), à droite par g(x₀)) ; différences finies centrées (pas h, puis h/2 pour l'estimation de Richardson de l'erreur) ; distances d_P
(au chemin droit de l'opérateur) et euclidienne (milieu du lien fin au segment grossier, image minimale). C'est un INSTRUMENT : aucune
conclusion physique dans ce lot. Spécification détaillée et seuils : avenant `v5.WP22` de GATES.md.

## 8. Hors périmètre

- GPU (noyaux APE/flot) : facultatif, seulement si A est vert et s'il reste du temps (gate RG-QL-GPU alors, par `gpu_t4.sh`).
- Groupes autres que SU(2) : non couverts (projection P et quaternions propres à SU(2)).
- Aucune lecture physique, aucune comparaison de β_eff entre opérateurs dans ce lot.

## 9. Sources (toutes ouvertes ; rien de cité de mémoire)

- Lüscher, arXiv:1006.4518 (JHEP 1008:071) : éq. (1.4) p. 2, (2.11)-(2.12) p. 4, phrase « mean-square radius … √8t » p. 4, (C.1) p. 19,
  (C.2)-(C.3) et remarque sur ε = 0,01 p. 20. Source LaTeX (archive arXiv `chantiers/scanner_irm_20261004/exploration/sources/1006.4518.src`,
  sha256 a5aaeb37fb604e01…) et PDF (sha256 63426411712d8b00…), images des pages 4, 19, 20 relues. `rageq` n'indexe pas ce papier
  (« papier inconnu ») : lecture directe, numéros contrôlés sur l'image.
- Greensite, hep-lat/0301023 (Prog. Part. Nucl. Phys. 51 (2003) 1) : éq. (55), p. 23, par `rageq` (arxiv_tex, confiance haute).
- BLOC2 (chantier vision, lecture seule) : `OPERATEURS.md`, `code/bloc2/operateurs.py`, `code/tests/test_jauge.py`, `VERDICT.md`.
- BILAN_VISION_2026-10-25 §38 et §40 ; `recherche/FLOT_GRADIENT_GR.md` (entrées 1, 2, 4, 10 : citations vérifiées par le chantier vision ;
  aucune équation n'en est reprise ici).
- Calculs : `validation/rg_ql/derivations_ql.py` (D1-D5, RC 0, `/usr/bin/python3 -I`, SymPy 1.14.0) ; `validation/rg_ql/support_py.py`.
