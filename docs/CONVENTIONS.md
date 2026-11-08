# Conventions du registre des groupes et valeurs dérivées (WP5, 2026-10-25)

Ce document fixe les conventions de `kvu groupe` et les valeurs de référence de G-SC et G-WC (SU(2)), **dérivées par plusieurs
canaux avant tout run** et gelées dans `GATES.md` v3.1. Règle anti-fabrication : aucune valeur ci-dessous n'est reprise sans second canal ;
quand un canal s'est trompé, l'erreur est écrite (section 6) au lieu d'être effacée.

## 1. Conventions

| objet | convention |
|---|---|
| lien | U_μ(x) ∈ G, de x vers x + e_μ ; SU(2) en quaternion Q = [a₀, a₁, a₂, a₃], U = a₀ + i(a₁σ₁ + a₂σ₂ + a₃σ₃), matrice [[a₀ + i a₃, a₂ + i a₁], [−a₂ + i a₁, a₀ − i a₃]] |
| action de Wilson | **S = β Σ_p (1 − Re χ_F(U_p)/d_F)**, somme sur les 6V plaquettes d'un réseau 4D périodique de volume V = L⁴ |
| observable | **P = ⟨(1/d_F) Re Tr U_p⟩** moyennée sur les plaquettes (SU(2) : ½ Tr, composante a₀ du produit en quaternion ; P = 1 à l'identité) ; c'est la première composante de `measure_quat` |
| générateurs | T^a hermitiens de trace nulle, [T^a, T^b] = i f^{abc} T^c, normalisés par **f^{acd} f^{bcd} = h∨ δ^{ab}** (racines longues de carré 2, C₂(adj) = h∨) |
| indice | Tr(T^a T^b) = T_F δ^{ab}, T_F = d_F C₂(F)/d_adj (mesuré par la fiche) |
| SU(2) | T^a = σ^a/2 ; f^{abc} = ε^{abc} ; d_F = 2, d_adj = 3, h∨ = 2, T_F = 1/2, C₂(F) = 3/4 ; réalité pseudoréelle (indicateur de Frobenius–Schur −1) |
| couplage | **β = d_F/(T_F g²)** ; SU(2) : β = 4/g² ; SU(N) : β = 2N/g² ; Sp(4) : β = 8/g² ; G₂ (en 7) : β = 7/g² |
| exponentielle | exp(a) = exp(i a^a T^a) ; SU(2) : (cos(|a|/2), sin(|a|/2) a/|a|) |
| logarithme | branche principale θ = atan2(\|v\|, a₀) ∈ [0, π] (demi-angle), a = 2θ v/\|v\| ; **ambiguë** si π − θ < 1e-6 ; **hors domaine** si θ > π/2 ; comptées par `CompteBranches` |
| RNG | Philox4x32-10 à compteur (`Flux`, étiquette `GROU` = 0x47524F55) ; échantillonneur de Haar SU(2) = `su2::cpu::haar_q` (validé G-HAAR) ; jamais de xorshift |

Les mutants de contrôle négatif (`Su2::avec_mutant`) ne servent qu'à la batterie G-MUT : `GenerateurRetire` (T³ retiré), `QrNaif`
(Gram–Schmidt d'une matrice gaussienne 2×2 : loi de Haar de U(2), pas de SU(2)).

## 2. G-SC : couplage fort (SU(2))

**Un degré de liberté.** Sous la densité de Weyl (2/π) sin²θ dθ sur [0, π], x = cos θ = ½ Tr U :
⟨x⟩ = 0, ⟨x²⟩ = 1/4, ⟨x⁴⟩ = 1/8 (SymPy). Le poids exp(βx) donne ⟨x⟩_β = I₂(β)/I₁(β) =: **u(β) = β/4 − β³/96 + β⁵/1536 − β⁷/23040 + …**
(développement SymPy ; quadrature à 60 chiffres = rapport de Bessel aux 30 chiffres imprimés). **Pente de tête = ⟨x²⟩ − ⟨x⟩² = 1/4 = 1/d_F².**

**Réseau 4D.** Développement en caractères e^{βx} = a₀[1 + Σ_j (a_j/a₀) χ_j], a_j = 2(2j+1) I_{2j+1}(β)/β ; coefficient fondamental
a_{1/2}/a₀ = 2u = d_F u. Les surfaces fermées contenant la plaquette observée avec toutes leurs faces en représentation fondamentale
sont les cubes élémentaires ; l'intégrale de Haar exacte du produit des 6 caractères d'un cube est **1/16** (contraction des tenseurs de
lien : 8 boucles d'indices, 2⁸/2¹² ; Monte Carlo de Haar 3·10⁶ tirages : 0,06282 ± 0,00058) ; une plaquette 4D est dans 4 cubes
(énumération). D'où ⟨χ_{1/2}⟩ = 2u + 4·(2u)⁵/16 + O(u⁷) et

**P = u + 4u⁵ + O(u⁷)** (les ordres u², u³, u⁴ n'existent pas ; le terme négligé est borné par 50u⁷ = 3,9·10⁻⁸ à β = 0,2, contre σ ≈ 4·10⁻⁵).
Cohérent avec les données archivées de (A) à β = 0,5 : P = 0,1238169(39) contre u + 4u⁵ = 0,1238339 (|z| = 0,37 avec la tolérance de A), alors que
u − 4u⁵ serait à ≈ 5σ.

**a_attendu.** L'ajustement exact aβ + bβ² aux points β₁ = 0,1 et β₂ = 0,2 donne a = (4 f₁ − f₂)/(2β₁) = 20 f₁ − 5 f₂, b = 50 (f₂ − 2 f₁), avec
f = u + 4u⁵. Valeurs (mpmath, 60 chiffres) : f(0,1) = 0,0249896288206509833…, f(0,2) = 0,0499181140897426339… ;

**a_attendu = 0,25020200596430649696** (b = −0,0030571775779666376). Le même ajustement sur u seul donne 0,25020742455852821 : le terme 4u⁵ décale a de 5,4·10⁻⁶,
négligeable devant σ_a ≈ 8·10⁻⁴ ; la pente de tête 1/4 diffère de a_attendu de 2,0·10⁻⁴ (biais du terme en β³ de u, que l'ajustement exact à deux points ne retire pas) : c'est
pourquoi a_attendu est celui de l'ajustement et non 1/4. Erreur : σ_a² = (16 σ₁² + σ₂²)/0,04 (chaînes indépendantes ; a = 20 P₁ − 5 P₂).

**Mutants.** Simulé à 1,05 β et analysé aux β nominaux : a = 0,2627331, soit +5,008 % (≈ 15 σ_a avec σ_a ≈ 8·10⁻⁴) ; à 1,002 β : +0,2003 % (information).

## 3. G-WC : couplage faible (SU(2))

À β → ∞, l'action s'écrit S ≈ (β T_F/(2 d_F)) Σ_p |F_p|² avec F_p le rotationnel de lien (SU(2) : (½)Tr exp(i x σ·n/2) = cos(x/2) = 1 − x²/8 + …) ; la
partie gaussienne vaut ⟨S⟩ = d_adj · rang(rot)/2 car chaque mode non nul apporte ½. Sur le tore L⁴ le rotationnel abélien (6V × 4V) a pour
rang **3V − 3** (vérifié par SVD : L = 3, V = 81, rang 240 ; L = 4, V = 256, rang 765) : V − 1 modes de jauge et 4 modes plats d'holonomie sont exclus.
Avec 6V plaquettes : ⟨1 − P⟩ = d_adj (3V − 3)/(2·6V β) = (d_adj/4β)(1 − 1/V).

**c₁ = lim β⟨1 − P⟩ = d_adj/4 = 3/4 (SU(2), V → ∞).** Correction gaussienne de volume fini : −1/V, soit −2,4·10⁻⁴ à L = 8 (V = 4096), 40 fois sous la tolérance de 2 %.
**Contrôle croisé SU(3) : c₁ = 8/4 = 2.** Mutant β × 1,05 : c₁_mesuré = c₁/1,05, soit −4,76 % (plus de deux fois la tolérance) ; β × 1,002 : −0,20 % (information).
Les modes plats (non gaussiens) apportent une correction O(1/V) du même ordre, absorbée par la tolérance et par c₂.

## 4. Protocoles (gelés dans GATES v3.1)

- **Estimateur.** Plaquette de chaque pas mesuré ; indices 1…N de `su2 gen` (le pas 0 de production, état à la fin de la thermalisation, est exclu) ; erreur √(2 τ var/N),
  var à N − 1, **τ à fenêtre automatique de Sokal, c = 6** (τ = ½ + Σ_{t ≤ W} ρ(t), W = premier t ≥ c τ(t)), comme `tests/stats.py` de (A).
- **G-SC.** L = 6, β ∈ {0,1 ; 0,2}, départ chaud, 1 000 pas de thermalisation, 2·10⁴ pas (1 HB + 4 OR), graines 26102605101/102 ; PASS ssi |a − a_attendu|/σ_a < 3 et |⟨P⟩ − (u + 4u⁵)|/σ < 3 pour chaque β.
- **G-WC.** L = 8, β ∈ {50, 100, 200, 400}, départ froid, 2 000 pas puis 2·10⁴ pas, graines 26102605201…204 ; y = β⟨1 − P⟩, σ_y = β σ_P, moindres carrés pondérés y = c₁ + c₂/β
  (c₁ = (S_xx S_y − S_x S_xy)/Δ, σ²(c₁) = S_xx/Δ, Δ = S S_xx − S_x², S_· sommes pondérées par 1/σ_y²) ; PASS ssi |c₁/c₁_att − 1| ≤ 0,02 et σ(c₁)/c₁ ≤ 0,005 ; la puissance non tenue donne SKIP (non vert).
- **G-HAAR.** 10⁶ tirages et graines 26102605001/002/003 ; z de ⟨χ⟩ = 0 et ⟨|χ|²⟩ = 1 avec erreur d'échantillon (|z| < 4) ; ⟨χ(U²)⟩ = −1 (SU(2)) à ± 4σ ; unitarité et déterminant ≤ 1e-13.
  `G-HAAR-3-CHAMP` (instrument ajouté par le lot) : mêmes seuils sur le champ d'un départ chaud L = 8 du backend (le noyau Haar CPU ou GPU lui-même), graine 26102605003.
- **G-UNIT.** L = 6, β = 2,3, départ chaud, graine 26102605301, 10⁴ pas (1 HB + 4 OR) ; max ||q| − 1| ≤ 1e-13 (F64, tout backend) ; ≤ 2e-6 (F32, GPU seulement : SKIP sur CPU, PASS exigé sur HIP/CUDA).
- **G-REG.** `A/tests/CRITERIA.md` T3 : L = 6, une chaîne chaude, β = 2,3 (graine 2510023) → 0,6029637(53) ; β = 2,5 (graine 2510025) → 0,6529010(31) ; 2 000 + 10 000 pas ; |z| < 3 avec l'erreur combinée.
- **G-MUT.** générateur retiré ⇒ G-ALG-1 ROUGE ; QR naïf ⇒ G-HAAR-2 et G-HAAR-3 ROUGES ; β × 1,05 ⇒ G-SC et G-WC ROUGES ; β × 1,002 en INFO. Un mutant VERT est un `ECHEC_OUTIL`.
- **G-CERT.** `kvu groupe gates --groupe su2 --certifier` écrit `validation/groupes/su2.json` (une entrée par backend : horodatage, sha256 du binaire, version, verdicts) et refuse de l'écrire
  si une ligne est FAIL ou SKIP (seul le SKIP prescrit `G-UNIT-F32` est admis dans l'entrée `cpu`, noté dans `skips_prescrits` : erratum v3.1 de GATES.md ; aucune entrée GPU n'admet de SKIP) ; `--groupe sp4|g2` est refusé (code 2) sans fiche ni certificat.

## 5. Canaux de dérivation et empreintes

| quantité | SymPy / mpmath (arbitre) | Haar numérique (échantillonneur validé de `kvu`) | Kimi K3.1 Pro en aveugle |
|---|---|---|---|
| pente 1/4 | exacte (⟨x²⟩ = 1/4) | ⟨x²⟩ = 0,249889 ± 2,5·10⁻⁴ (10⁶ tirages, graine 26102605401) | tour 1 : **1/2 (faux)** ; tour 2 : **1/4 (accord)**, coefficient de β³ = −1/96, u(0,1), u(0,2) à 20 chiffres identiques à SymPy |
| u(β) | I₂/I₁ = quadrature à 60 chiffres | u(0,1) = 0,02467 ± 5·10⁻⁴, u(0,2) = 0,04959 ± 5·10⁻⁴ (attendus 0,024990, 0,049917 : −0,64σ, −0,66σ) | tour 1 : β/2 − β³/16 (faux) |
| c₅ = +4 | contraction exacte du cube : 1/16 ; 4 cubes (`wp5_cube.py`) | Monte Carlo du cube : +0,06282 ± 0,00058 | tour 1 : **−4 (faux)** ; tour 2 : **+4 (accord)** : cube = 1/16 (8 boucles), 4 cubes, Monte Carlo +0,06220 ± 0,00071 |
| a_attendu | 0,25020200596430649696 | a (sur u seul) = 0,2455 ± 0,010 (−0,46σ de 0,2502074) | tour 1 : 0,2502128 (de P = u − 4u⁵) |
| c₁ | 3/4 (1 − 1/V) ; rang 3V − 3 vérifié ; SU(3) : 2 | — | 3/4, correction (1 − 1/V), SU(3) : 2 (**accord**) |
| T_F, h∨, β(g²) | 1/2, 2, 4/g² | — | 1/2, 2, 4/g² (**accord**) |

Empreintes (sha256, 16 premiers caractères) : `scratch/wp5/sympy/wp5_sympy.py` 2ebab887…, sortie `wp5_sympy.out` e161a5ec… ; `wp5_cube.py` ca57cb4a…, sortie `wp5_cube.out` d5223288… ;
prompt Kimi 1 133d4e00…, réponse 1 a3bd718a… ; prompt Kimi 2 c3b9e79d… ; réponse 2 8c93eaae… (tour 2 : 5 résultats A, B1-B4, tous égaux à ceux de SymPy).

## 6. Erreurs relevées pendant la dérivation

Kimi, tour 1, s'est trompé deux fois : (a) le développement de I₂/I₁ (β/2 au lieu de β/4 ; l'erreur n'a pas d'effet sur le rapport lui-même, qu'il donne juste) ; (b) le signe du terme de cube (−4u⁵
au lieu de +4u⁵, « chaque cube contribue par −u⁵ » sans calcul), d'où a = 0,2502128 au lieu de 0,2502020. Les deux sont tranchées par SymPy (quadrature, contraction exacte du cube),
par l'intégration de Haar numérique et par les données archivées de (A) à β = 0,5. La règle du brief (désaccord ⇒ arrêt) a été appliquée : aucun run de G-SC/G-WC n'a eu lieu
avant le second tour de Kimi. Au tour 2 (questions étroites, vérification numérique exigée, toujours sans valeur attendue montrée), Kimi retrouve la pente 1/4, le cube 1/16 (après avoir
corrigé l'orientation de trois faces dans son propre code), 4 cubes et c₅ = +4 : **les trois canaux s'accordent** sur toutes les valeurs de v3.1. Son tour 1 reste consigné comme faux.
Rappel de méthode : l'accord de deux modèles n'est pas une preuve ; ce sont SymPy, la contraction exacte et l'intégration de Haar numérique qui tranchent.

## 7. Sp(4) au registre (WP16, tranche 3, 2026-10-26)

Valeurs gelées dans `GATES.md` v4.1 (valeurs) et **v4.4** (protocole d'exécution, relu quatre fois par GPT-6.1-Sol en lecture seule avant le hachage ; réponses `scratch/wp16/sol_rev1..4.log`). Dérivations du prélude (`scratch/t3-prelude/sympy/`,
SymPy/mpmath exécutés, Haar numérique indépendant, Kimi K3.1 Pro en aveugle : accord sur toutes les valeurs ci-dessous) ; aucun chiffre n'est repris d'un raisonnement.

| objet | convention |
|---|---|
| groupe | USp(4) = {U ∈ U(4) : UᵀJU = J}, J = [[0, I₂], [−I₂, 0]] ; d_F = 4, d_adj = 10, pseudoréel (Frobenius–Schur −1), h∨ = 3, C₂(F) = 5/4, T_F = ½ |
| algèbre | A hermitienne avec AᵀJ + JA = 0 ⇔ A = [[P, Q], [Q̄, −P̄]], P hermitienne 2×2, Q complexe symétrique 2×2 (dim 4 + 6 = 10) |
| générateurs | c = 1/(2√2), σ₀ = I : a = 0…3 : P = c σ_a ; a = 4, 5 : Q = ½E₁₁, i·½E₁₁ ; a = 6, 7 : Q = ½E₂₂, i·½E₂₂ ; a = 8, 9 : Q = c(E₁₂ + E₂₁), i·c(E₁₂ + E₂₁) ; Tr(T^aT^b) = ½δ^{ab} **une fois pour toutes** (la fiche ne renormalise jamais) |
| action | S = β Σ_p (1 − Re Tr U_p/4), β = d_F/(T_F g²) = 8/g² (même action que le moteur (C), `beta_norm = d_fund`) |
| exponentielle | exp(i Σ c_a T^a) en forme fermée (Cayley–Hamilton : A a les valeurs propres ±λ₁, ±λ₂), différences divisées stabilisées ; référence de test = Taylor avec mise à l'échelle |
| logarithme | principal, branche de norme minimale, θ = plus grand |phase propre| ∈ [0, π] ; ambiguë si π − θ < 1e-6 ; hors domaine si θ > π/2 ; comptées par `CompteBranches` |
| projection | `projeter` = J-symétrisation (U = J Ū J⁻¹) puis facteur polaire par Newton ; dans le moteur, la ré-unitarisation symplectique par Gram–Schmidt (colonne 0, colonne 1 ⟂ {c₀, −J c̄₀}, colonnes 2, 3 = −J conj(colonnes 0, 1)) |

**Haar (choix (b) du brief : QR quaternionique, USp(4) = U(2, ℍ)).** Colonne 0 gaussienne complexe de ℂ⁴ normalisée ; colonne 2 = −J conj(colonne 0) ; colonne 1 = gaussienne orthogonalisée à c₀ et c₂ puis normalisée ; colonne 3 = −J conj(colonne 1).
Argument : l'application v ↦ −J v̄ est antiunitaire et commute avec l'action de USp(4) ; la loi de c₀ (gaussienne isotrope normalisée) est invariante par USp(4) (transitif sur S⁷), et sachant c₀ la loi de c₁ est
invariante par le stabilisateur Sp(1) (transitif sur S³ du complément quaternionique) : la loi du produit est invariante à gauche, donc Haar. **Le choix est jugé seulement par les gates** (G-HAAR-1/2/3, G-HAAR-3-CHAMP). Le **QR naïf** de (C)
(Gram–Schmidt de 4 colonnes gaussiennes complexes sans contrainte) est la mesure de Haar de U(4) : ⟨χ(U²)⟩ → 0 au lieu de −1, |UᵀJU − J| ~ 2 ; c'est le mutant de G-MUT-HAAR2/3 (registre 1711).

**Série de couplage fort.** u(β) = ⟨(χ/4)e^{βχ/4}⟩/⟨e^{βχ/4}⟩, χ = Re Tr U, recalculée dans le code à 40 termes à partir des moments exacts de Weyl m_k (formule d'intégration de Weyl, densité (cosθ₁ − cosθ₂)² sin²θ₁ sin²θ₂, règle des
trapèzes à 128 points par angle, exacte en degré < 128) ; m = (1, 0, 1, 0, 3, 0, 14, 0, 84, …) retrouvés à 1e-14, coefficients de Taylor (1/16, 0, 0) à 3e-16 (seuil 1e-12), a recalculé = a_attendu gelé à 3e-16
(`G-SC-SERIE`, instrument ajouté). P = u + 4u⁵ (v4.1) ; **a_attendu = 0,06249999750641898525**, c₁_attendu = 5/2.

**Pas Monte Carlo** (v4.0, v4.4.1) : 10 propositions Metropolis successives par lien (ε₀ = 0,3 adapté par blocs de 50 pas **pendant la thermalisation seulement**, cible 0,4–0,6), puis un balayage de sur-relaxation (Brown–Woch) sur les sous-groupes SU(2) longs
(e₀, e₂) puis (e₁, e₃) : n = (Re p, −Im p, Re q, −Im q), p = W_ii + conj(W_jj), q = W_ji − conj(W_ij), r′ = n̂² (réflexion microcanonique exacte, involutive localement). RNG Philox à compteur [pas, lien, tirage, étiquette].

**Lecture des chaînes** (v4.4.2) : séries de plaquettes (une valeur par pas) ; Sokal c = 6 pour G-SC/G-WC ; G-REG par jackknife entre 16 chaînes et seuil de Hsu (quantile de Student ν = 9 calculé dans le code : 4,0942553…) ; extension unique ligne par ligne ;
une ligne SKIP n'affiche aucune valeur physique. **Interprétation consignée** : « erreur nulle » (v4.4.2-5bis) est lue comme nulle à la précision machine (σ ≤ 1e-14 |⟨P⟩|), parce qu'une série constante donne σ ~ 1e-17 par arrondi.
Des gates vertes établissent l'absence d'écart détecté à leur précision, pas l'absence de biais stationnaire (v4.4.2-5ter) : la ré-unitarisation symplectique après chaque lien est une perturbation numérique (≈ 1e-16), jamais supposée sans effet.

**Limites déclarées** : G-WC ne détecte pas un biais < 2 % (v3.4) ; G-REG compare à un moteur (C) dont le Haar était faux (un FAIL ouvre un diagnostic sans dire quel code se trompe) ; volume 16 contre 32 non corrigé.
