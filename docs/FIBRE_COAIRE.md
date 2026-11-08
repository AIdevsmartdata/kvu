# Fibre de Bałaban : jacobien de la coaire et cas exacts (WP11)

- **Lot** : WP11 de la tranche 2 du programme `kvu` (PLAN_TRAVAIL §2), branche `wp11`, agent Opus, 2026-10-25/26.
- **Périmètre** : mathématiques seulement, aucun code Rust. Ce document et le dossier `crates/kvu/validation/fibre_coaire/`
  (scripts SymPy, mpmath et numpy, sorties, dérivation de Kimi).
- **Statut** : les quatre résultats demandés sont dérivés, vérifiés mécaniquement et confrontés à une seconde dérivation en aveugle
  (Kimi K3.1 Pro). Gate **FIB-DERIV : PASS** (§9). Les trois désaccords sont tranchés par SymPy ou par le calcul (§7).
- Rien n'est gelé dans `GATES.md` par ce lot. Les gates proposées au §9 sont à geler, si le pilote le décide, dans une version v4.

## 0. Ce qu'il faut retenir (pour WP12a et WP12b)

1. **Mesure conditionnelle.** μ(dU | Ū = V) = Z(V)⁻¹ e^{−S(U)} det(J Jᵀ)^{−1/2} 𝓗^{3(N−n)}(dU) sur la fibre.
   - J = DŪ en repères orthonormés ; 𝓗 est la mesure riemannienne (de Hausdorff) de la fibre.
   - La marginale grossière vaut ρ̄(V) = (2π²)^{n−N} Z(V) (§2).
   - Le facteur det(J Jᵀ)^{−1/2} est **invariant de jauge** et ne dépend pas de la trivialisation.
2. **Jacobien de la carte de Bałaban en forme fermée**, avec les dérivées de log et exp sur SU(2) (§3).
   - Vérifié par différences finies : 3·10⁻²⁸ sur les jouets (gate : 10⁻⁸) et 1,8·10⁻¹¹ sur la vraie carte (432 blocs 3×3).
   - L'implémentation de référence (`balaban_np.py`) redonne les fixtures de WP6 à 2,9·10⁻¹⁵.
3. **Route B (RATTLE).** Le potentiel correct est **S + ½ log det(J M⁻¹ Jᵀ)** (M = 1 : S + ½ log det(J Jᵀ), comme ARCHITECTURE §5).
   - Kimi soutenait le signe opposé, et « rien à ajouter pour M = 1 ».
   - Une expérience RATTLE la réfute : écart de 278 à 426 σ ; la prédiction de Liouville tient à |z| ≤ 1,4 sur les 6 runs (§7, D1).
4. **Route A (pénalité) : le paramètre est κ = β ε².**
   - Avec les ε prévus (0,2 ; 0,1 ; 0,05) à β = 8, le biais exact du cas gaussien est **non linéaire** en ε².
     Il vaut +20 %, +15 % et +8 % sur la variance de la boucle 3×3 des centres, et +1,0 %, +0,8 % et +0,4 % sur ⟨S⟩.
   - Proposition : **ε = (κ/β)^{1/2}, κ ∈ {10⁻³ ; 2·10⁻³ ; 4·10⁻³}**, régime linéaire (§2.5, §5.4).
5. **Le mutant « sans jacobien » est faible sur réseau.** À l'ordre dominant, son effet sur ⟨1 − P⟩ (6⁴, V = 1) vaut
   **−0,029/β** en relatif : −0,003 % à β = 1000, −0,36 % à β = 8, ≈ −1 % à β = 2,8 (§6.3).
   - FIB-GAUSS (β = 1000, tolérance 3 %) **ne peut pas** le voir.
   - FIB-MUT doit donc se juger sur le jouet à deux liens (effet +49 % à β = 2) et sur FIB-AB à β = 2,8, avec une statistique
     dimensionnée sur ces nombres (§9).
6. **U(1).** Le jacobien y est constant : U(1) teste la contrainte et la mesure de fibre, **pas** le facteur de coaire.
   - En U(1) compact, la coupure du log découpe la fibre en cellules (codimension 1), et la route B n'y est pas ergodique.
   - FIB-U1 doit utiliser l'U(1) **gaussien non compact**, dont la loi conditionnelle est écrite exactement (§5).
   - Le vecteur de 10 moments attendu par le brouillon v4-T2 est livré exact, et aussi pour la route A à chaque ε (§5.5).
7. **Brouillon GATES v4-T2 partie 1.** FIB-AB (choix de ε, ajustement en ε²) et FIB-MUT (puissance) y sont en tension avec les
   points 4 et 5 ci-dessus (§9, note finale).

## 1. Conventions

| objet | convention (vérifiée) |
|---|---|
| élément de SU(2) | W = exp(iω·σ) = cos θ + i sin θ (n·σ), θ = \|ω\| ∈ [0, π], n = ω/θ (θ est le **demi-angle**). Quaternion kvu [a₀, a₁, a₂, a₃] = [cos θ, sin θ n] (`docs/CONVENTIONS.md` §1) |
| logarithme | branche principale ω(W) = θn, définie et analytique sur SU(2) ∖ {−1}. C'est `rg::bloc::log_q`, et `exp_q(s) = exp(i s·σ)` |
| autres coordonnées | a = 2ω (WP5 : exp(i aᵃTᵃ), Tᵃ = σᵃ/2) ; x = √2 ω (S1/Q3 `q3_lib`, `ym4_hess` : T_a = iσ_a/√2) |
| tangent | perturbation **à gauche** U → exp(iε·σ)U. Les coordonnées ε sont orthonormées pour la métrique bi-invariante retenue : SU(2) est alors la sphère S³ de rayon 1, de volume 2π² |
| Haar | dU = dvol/(2π²) ; en coordonnées exponentielles, dU = (2π²)⁻¹ (sin θ/θ)² d³ω |
| adjointe | R(A)_{ab} = ½ Re Tr(σ_a A σ_b A†), A (iε·σ) A† = i (R(A)ε)·σ. R(exp(iθn·σ)) est la rotation d'angle −2θ autour de n ; en quaternion, R = (a₀² − \|v\|²)1 − 2a₀[v×] + 2vvᵀ |
| chemin | Γ = (ℓ₁^{s₁}, …, ℓ_m^{s_m}), U(Γ) = U_{ℓ₁}^{s₁} ⋯ U_{ℓ_m}^{s_m} (ordre du chemin ; s = −1 : U†) |
| carte de Bałaban (WP6) | Ū_c = U₀ exp(Σ_Γ w_Γ log(U₀⁻¹ U(Γ))), U₀ = U(Γ₀). Lien grossier c = (z, μ), centre 3z + (1,1,1,1), Γ₀ = segment droit c → c + 3e_μ, Γ = P·[x, x + 3e_μ]·P′⁻¹, w = 1/(81 \|G(c,x)\|²), soit **10 473 chemins** (`balaban_np.gabarit`, contrôlé par FIB-CARTE-CHEMINS) |
| action | Wilson, S = β Σ_p (1 − ½ Tr U_p) |
| jacobien | J = DŪ, matrice 3n × 3N de blocs J_{cℓ} : si U_ℓ → e^{iε_ℓ·σ}U_ℓ, alors Ū_c → e^{iΔ_c·σ}Ū_c avec Δ_c = Σ_ℓ J_{cℓ} ε_ℓ + O(ε²) |

**Lien avec Bałaban.** La carte est la moyenne non linéaire de CMP 109, réécrite par conjugaison
(`U₀ exp(Σw log(U₀⁻¹U_Γ)) = exp(Σw log(U_Γ U₀⁻¹)) U₀`, S1/VERDICT §2.1) :
- [lecture] Bałaban, CMP 109 (1987) 249-301, (0.4), p. 253 — OCR GLM, désaccord GLM/Ovis — `reading_list/equations/crops/22597.png`
  (rageq id 22597) — vérifié sur l'image : oui. Le texte lu est
  Ū(c) = exp[ i Σ_{x∈B(c₋)} L^{−d} Σ_{Γ∈G(c₋,x)} |G(c₋,x)|⁻¹ Σ_{Γ′∈G(c₊,x′)} |G(c₊,x′)|⁻¹ (1/i) log U(Γ ∪ [x,x′] ∪ (−Γ′) ∪ (−c)) ] U(c).
- [lecture] même article, (0.3), p. 252 (blocs B^k(y)) — OCR GLM — `crops/22595.png` (id 22595) — vérifié sur l'image : oui.

## 2. Désintégration de Haar ⊗ e^{−S} par Ū (formule de la coaire)

### 2.1 Énoncé

Notations :
- 𝒟 = {U ∈ SU(2)^N : U₀(c)⁻¹U(Γ) ≠ −1 pour tout c et tout Γ}.
  - La coupure du log principal sur SU(2) est le **seul point −1**, donc chaque condition retire un ensemble de codimension 3.
  - 𝒟 est donc ouvert, connexe et de mesure pleine, et Ū y est analytique réelle.
- 𝒟* = {U ∈ 𝒟 : rang J(U) = 3n}.
  - J(1) = Q ⊗ 1₃ est de rang 3n, car rang Q = n (FIB-U1-Q : 64 = n à L = 6).
  - L'ensemble analytique {rang < 3n} est donc propre dans 𝒟 connexe, donc négligeable.
- Pour V valeur régulière (presque tout V, par Sard), F_V = Ū⁻¹(V) ∩ 𝒟* est une sous-variété lisse de dimension 3(N − n).

**Énoncé.** Pour toute F mesurable positive,

    ∫ F(U) e^{−S(U)} dU = ∫_{SU(2)^n} ρ̄(V) ⟨F⟩_V dV,
    μ(dU | Ū = V) = Z(V)⁻¹ e^{−S(U)} det(J(U) J(U)ᵀ)^{−1/2} 𝓗^{3(N−n)}|_{F_V}(dU),
    Z(V) = ∫_{F_V} e^{−S} det(J Jᵀ)^{−1/2} d𝓗,
    ρ̄(V) = (2π²)^{n−N} Z(V).

Le δ formel de l'architecture se lit donc :

    ∫ F e^{−S} ∏_c δ(Ū_c V_c⁻¹) dU = (2π²)^{n−N} ∫_{F_V} F e^{−S} det(J Jᵀ)^{−1/2} d𝓗,

où δ est la masse de Dirac à l'identité relative à la mesure de Haar de SU(2).

**Preuve (esquisse).** On applique la formule de la coaire pour une submersion entre variétés riemanniennes
(Federer, *Geometric Measure Theory*, Springer 1969, réimpr. 1996, chap. 3 « Rectifiability », §3.2, DOI 10.1007/978-3-642-62010-2_4).
1. En U ∈ 𝒟*, la restriction de J à (ker J)^⊥ est un isomorphisme sur T_V SU(2)^n.
   La valeur absolue de son déterminant (repères orthonormés) vaut det(J Jᵀ)^{1/2}.
2. Dans une carte produit (théorème de submersion), dvol = d𝓗_fibre ∧ Ū*(dvol)/det(J Jᵀ)^{1/2}.
3. Les constantes viennent de dU = dvol/(2π²)^N et dV = dvol/(2π²)^n.

Vérification exacte sur le jouet (§4) : √(det g_fibre)·det(J Jᵀ)^{−1/2} = sin²θ/θ², exactement la densité de Haar de W (SymPy, et 60 points à 10⁻³¹).

### 2.2 Invariances (vérifiées)

- **Trivialisation.** Passer à droite sur un lien source multiplie la colonne de blocs par R(U_ℓ) ; passer à droite au but
  multiplie la ligne par une rotation. Les deux sont orthogonaux, donc det(J Jᵀ) est inchangé
  (FIB-TRIVIALISATION-B/C : 7·10⁻⁵⁰ relatif).
- **Échelle de métrique.** Avec la même échelle sur toutes les sources et tous les buts (par exemple la métrique de S1, double de
  la nôtre), J est inchangé et 𝓗 est multiplié par une constante : μ(·|V) ne change pas.
- **Jauge.** Ū(g·U) = g(c) Ū g(c′)† donne J(g·U) = R(g(c)) J(U) O_g⁻¹, avec O_g orthogonale (action de jauge sur les liens).
  Donc det(J Jᵀ)(g·U) = det(J Jᵀ)(U) pour **toute** jauge, et pas seulement celles qui fixent les centres.
  Vérifié sur 6⁴ par FIB-BINF-JAUGE : jauge de Haar quelconque en chaque site, log det = 42,333479311690 dans les deux cas,
  écart 2,8·10⁻¹⁴.

### 2.3 Ce que Bałaban intègre réellement

L'intégrale de (0.17) contient, en plus de ∏_c δ(Ū(c)V⁻¹(c)) dU et du poids d'action :
- χ₀, la restriction aux petits champs ;
- un facteur de fixation de jauge exponentiel exp[−g₀⁻² Σ_{y∈T^{(1)}} Σ_{x∈B(y), x≠y} (1 − Re tr U(y,x))].

Références :
- [lecture] Bałaban, CMP 109 (0.17), p. 255 (p. 7 du PDF) — OCR GLM (rageq id 22610) — vérifié sur l'image (page rendue par pdftoppm) : oui ;
- (0.15)-(0.16), p. 254-255 (ids 22608, 22609) — vérifié sur l'image : oui. Le texte dit que l'intégrale sur les transformations
  de jauge u telles que u(y) = 1 pour y ∈ T^{(1)} « is equal to 1 » : le facteur de jauge est normalisé par z.

Conséquences :
- Pour toute observable invariante sous ces transformations résiduelles, le facteur de fixation de jauge ne modifie **pas** μ(·|V).
- Les observables invariantes de jauge de WP12 (⟨1 − P⟩, boucles de Wilson, spectre d'opérateurs covariants) peuvent donc ignorer
  la fixation de jauge de Bałaban.
- χ₀ n'est pas dans la fibre de `kvu` : le domaine est suivi par RG-DOMAINE.
- Une observable non invariante de jauge (Hessienne en tranche, WP13) demande en revanche un traitement de Faddeev–Popov
  explicite, **hors périmètre** ici.

### 2.4 Route B (RATTLE) : quel potentiel ?

Cadre :
- dynamique hamiltonienne contrainte sur la fibre ;
- énergie cinétique ½ pᵀM⁻¹p, M constante dans les repères orthonormés ;
- impulsions tirées dans N(0, M), puis projetées M⁻¹-orthogonalement sur {J M⁻¹ p = 0}.

Cette dynamique conserve la mesure de Liouville de T*F_V. Sa projection sur les positions est e^{−V} σ_M, le volume riemannien de
la fibre pour la métrique M. Voir Lelièvre, Rousset, Stoltz, *Hybrid Monte Carlo methods for sampling probability measures on
submanifolds*, Numer. Math. 143 (2019) 379-421, DOI 10.1007/s00211-019-01056-4 (vérifié par Crossref ; correctif :
Numer. Math. 144 (2019) 447-449, DOI 10.1007/s00211-019-01086-y).

Dans les coordonnées q′ = M^{1/2}q, la métrique cinétique devient euclidienne et J′ = J M^{−1/2}. Pour cibler μ(·|V), il faut donc

    V = S + ½ log det(J M⁻¹ Jᵀ)   (+ constante) ;   M = 1 : V = S + ½ log det(J Jᵀ).

C'est la prescription d'ARCHITECTURE §5 et de PLAN WP12a. Elle est arbitrée contre la thèse de Kimi au §7 (D1) : SymPy et
6 runs RATTLE.

La force contient ∂J, qui demande les dérivées secondes de Ū. Deux options :
- les nombres duaux (mode direct) sur la formule fermée du §3.3 ;
- les dérivées fermées de L_g et E_g du §3.2, avec ∂R(P)/∂ε_ℓ = −2[(T_P(ℓ)ε) ×] R(P), où T_P est le transport du §3.3
  (pour P = A U_ℓ B, T_P(ℓ) = R(A)).

### 2.5 Route A (pénalité) : convergence et biais

Par le §2.1, la mesure pénalisée μ_ε ∝ e^{−S − Σ_c|ω(V_c⁻¹Ū_c)|²/(2ε²)} dU s'écrit ∫ dV′ ρ̄(V′) k_ε(V′V⁻¹) μ(·|V′).
C'est un mélange des lois conditionnelles sur un voisinage de taille ε de V. Elle converge vers μ(·|V) quand ε → 0, et le
facteur de coaire est inclus **automatiquement**.

Le biais vient de la variation de ρ̄(V′)⟨O⟩_{V′} à l'échelle ε. Aux grands β, ρ̄ ∝ e^{−β φ̄ᵀM_c φ̄}, donc :
- le paramètre est **κ = β ε²** ;
- la pénalité ne domine un mode grossier de raideur λ que si 2κλ ≪ 1 ;
- or λ_max(M_c) = **51,62** à L = 6 (§5.5).

Le §5.4 donne la loi exacte et la table du biais. La règle qui en sort : **κ ≤ 4·10⁻³** pour être dans le régime linéaire.

## 3. Dérivées fermées sur SU(2) et jacobien de la carte

### 3.1 log et exp

On note W = e^{iω·σ}, θ = |ω|, n = ω/θ, P = nnᵀ.

    L_g(ω) = P + θ cot θ (1 − P) + [ω×]            log(e^{iη·σ}W) = i(ω + L_g η + O(η²))·σ
    L_d(ω) = P + θ cot θ (1 − P) − [ω×] = L_g(−ω)   log(W e^{iη·σ}) = i(ω + L_d η + O(η²))·σ
    E_g(ω) = P + (sin θ cos θ/θ)(1 − P) − (sin²θ/θ²)[ω×] = L_g⁻¹    e^{i(ω+δ)·σ} = e^{i(E_g δ)·σ} e^{iω·σ}
    E_d(ω) = P + (sin θ cos θ/θ)(1 − P) + (sin²θ/θ²)[ω×] = L_d⁻¹    e^{i(ω+δ)·σ} = e^{iω·σ} e^{i(E_d δ)·σ}

Formes équivalentes (celles de Kimi) : L_g = 1 + X + (1/θ² − cot θ/θ) X² et E_g = 1 − (sin²θ/θ²) X + ((θ − sin θ cos θ)/θ³) X², avec X = [ω×] et X² = θ²(P − 1).
L_g est g(ad_X) avec g(z) = z/(e^z − 1) et ad ↔ −2θ[n×].

Limites :
- θ → 0 : L_g = 1 + [ω×] + ⅓[ω×]² + O(θ⁴), conforme à BCH.
- θ → π : θ cot θ ∼ −π/(π − θ). La partie transverse de L diverge, c'est la coupure en W = −1.
- θ → π : E_g → nnᵀ (désaccord D2 tranché).

Vérifications (`t1_dlog.py`) :
- SymPy exact : DLOG-INVERSE, DLOG-BERNOULLI, DLOG-BCH, DLOG-PI, DLOG-EG-PI ;
- différences finies du log générique, par diagonalisation, à 60 chiffres : 3·10⁻³⁴ sur 24 points, θ ∈ [10⁻⁶, π − 10⁻³].

**Piège noté.** `mpmath.logm` n'est pas un log de référence fiable près de −1 : à θ = π − 10⁻², il renvoie un logarithme de trace
2πi, la valeur propre e^{−iθ} étant prise sur l'autre feuillet. Les différences finies passent par `mp.eig`.

### 3.2 Dérivées de L_g et E_g (force de la route B)

On écrit L_g = f 1 + h ωωᵀ + [ω×] et E_g = s 1 + k ωωᵀ − m [ω×], avec
f = θ cot θ, h = (1 − f)/θ², s = sin 2θ/(2θ), k = (1 − s)/θ², m = sin²θ/θ². Alors

    ∂L_g/∂ω_k = (f′/θ) ω_k 1 + (h′/θ) ω_k ωωᵀ + h (e_k ωᵀ + ω e_kᵀ) + [e_k ×]
    ∂E_g/∂ω_k = (s′/θ) ω_k 1 + (k′/θ) ω_k ωωᵀ + k (e_k ωᵀ + ω e_kᵀ) − (m′/θ) ω_k [ω×] − m [e_k ×]

Les limites en θ → 0 de f′/θ, h′/θ, s′/θ, k′/θ et m′/θ valent −2/3, 2/45, −4/3, −4/15 et −2/3 (SymPy) : les dérivées sont
régulières en ω = 0. Différences finies : 2·10⁻³⁴ (DLOG-DERIVEES).

### 3.3 Jacobien de la carte de Bałaban

**Transport le long d'un chemin.** Pour un lien fin ℓ,

    T_Γ(ℓ) = Σ_{k : ℓ_k = ℓ} s_k R(P*_k),

où P*_k est le transport du début de Γ jusqu'à la **queue** de ℓ_k le long de Γ :
- si s_k = +1, c'est le produit des facteurs qui précèdent ;
- si s_k = −1, c'est ce produit **multiplié par U_ℓ⁻¹**.
Une perturbation à gauche de ℓ donne ainsi U(Γ) → e^{i T_Γ(ℓ)ε·σ} U(Γ).

**Jacobien.** Avec ω_Γ = ω(U₀⁻¹U(Γ)), ω̄ = Σ w_Γ ω_Γ, R₀ = R(U₀) et T₀ = T_{Γ₀},

    J_{cℓ} = (1 − B_tot) T₀(ℓ) + Σ_Γ B_Γ T_Γ(ℓ),    B_Γ = w_Γ R₀ E_g(ω̄) L_g(ω_Γ) R₀ᵀ,    B_tot = Σ_Γ B_Γ.

**Dérivation.**
1. W_Γ = U₀⁻¹U(Γ) reçoit la perturbation gauche R₀ᵀ(T_Γ − T₀)ε.
2. Donc δω_Γ = L_g(ω_Γ) R₀ᵀ(T_Γ − T₀)ε.
3. Ū = U₀ e^{iω̄·σ} reçoit à gauche T₀ε + R₀ E_g(ω̄) δω̄.

La formule couvre les répétitions et les parcours à rebours. En U = 1, elle se réduit à J = Q ⊗ 1₃.

| vérification | résultat | script |
|---|---|---|
| jouet à 2 liens, 30 configurations de Haar, t ∈ {½ ; 0,3 ; 0,8} | 3,6·10⁻²⁸ (gate 10⁻⁸) | `t2_jouet.py` |
| jouets B (2 liens, 4 chemins) et C (3 liens, 2 liens grossiers), avec rebours et répétitions | 2,2·10⁻²⁸ ; 1,7·10⁻²⁸ | `t3_chemins.py` |
| vraie carte L_b = 3, fixtures 9⁴ de WP6 (amplitude 0,6, θ max 3,13) | Ū = fixtures à 2,9·10⁻¹⁵ ; J = DF (Richardson) à 1,8·10⁻¹¹ sur 432 blocs | `t4_balaban.py` |

## 4. Bloc jouet à deux liens (cas exact)

Définition : Ū = U₁ exp(t log(U₁⁻¹U₂)), t ∈ (0, 1), W = U₁⁻¹U₂ = e^{iω·σ}.

**Jacobien.** J = [1 − A, A], avec A = t R(U₁) E_g(tω) L_g(ω) R(U₁)ᵀ, et

    det(J Jᵀ) = ((1 − t)² + t²) · D_t(θ)²,   D_t = (sin²(tθ) + sin²((1−t)θ))/sin²θ.

Pour t = ½ : det(J Jᵀ) = 1/(8 cos⁴(θ/2)).

**Fibre.** U₁ = V e^{−itω·σ}, U₂ = V e^{i(1−t)ω·σ}, |ω| < π. Sa densité de volume vaut
√det g = √((1−t)² + t²) (sin²(tθ) + sin²((1−t)θ))/θ².

**Coaire.** √det g · det(J Jᵀ)^{−1/2} = sin²θ/θ². Donc **W | Ū = V suit la loi de Haar, pour tout t et tout V**.
Argument direct : (U₁, U₂) ↦ (Ū, W) préserve Haar ⊗ Haar. Contrôle Monte Carlo : 2·10⁵ tirages, ⟨x_W⟩ = 0,0008 ± 0,0011,
⟨x_W²⟩ = 0,25006 ± 0,00056, corrélations avec Ū nulles à ± 0,002 (INFO).

**Lois avec action.** Pour S = −β ½Tr(U₁⁻¹U₂), on pose x = ½ Tr W = cos θ.
- Loi correcte : ⟨x⟩ = I₂(β)/I₁(β) et ⟨x²⟩ = (I₃ + I₂/β)/I₁. Deux canaux, Bessel et quadrature, accord à 10⁻³¹.
- **Mutant** sans facteur de coaire : densité radiale e^{β cos θ}(sin²(tθ) + sin²((1−t)θ)) ;
  pour t = ½, ⟨x⟩ = (2I₁ − I₀ − I₂)/(2(I₀ − I₁)).
- Potentiel de la route B : ½ log det(J Jᵀ) = ½ log((1−t)² + t²) + log D_t(θ), soit −½ log 8 − 2 log cos(θ/2) pour t = ½.
- **U(1)** (θ₁, θ₂ ; φ̄ = θ₁ + t[θ₂ − θ₁]) : J = [1 − t, t] est constant, ⟨cos ψ⟩ = I₁/I₀, et le mutant est identique.

Valeurs gelées (`sorties/valeurs_jouet.json`, mpmath à 30 chiffres) :

| β | ⟨x⟩ = I₂/I₁ | ⟨x²⟩ | mutant t = ½ : ⟨x⟩ | mutant t = 0,3 : ⟨x⟩ | effet relatif du mutant sur ⟨1 − x⟩ (t = ½) | U(1) : ⟨cos ψ⟩ = I₁/I₀ |
|---|---|---|---|---|---|---|
| 2 | 0,433127426722312 | 0,350308859916532 | 0,154394686533120 | 0,222907311718793 | +49,17 % | 0,697774657964008 |
| 2,8 | 0,545098533322141 | 0,415965857154849 | 0,376887225945948 | 0,420582423877691 | +36,98 % | 0,794038834374382 |
| 8 | 0,819249410355621 | 0,692781471116642 | 0,805069521286610 | 0,808484173193561 | +7,85 % | 0,935235493529439 |
| 1000 | 0,998500375375493 | 0,997004498873874 | 0,998499624436301 | 0,998499802596186 | +0,050 % | 0,999499874874804 |

## 5. Cas U(1) exact

### 5.1 Carte et jacobien

Liens diagonaux U_ℓ = e^{iθ_ℓσ₃}. La carte SU(2) restreinte au sous-groupe diagonal **est** la carte U(1) :

    φ̄_c = (Qθ)_c − 2π Σ_Γ w_Γ n_Γ(θ)   (mod 2π),

où n_Γ ∈ ℤ est l'entier de branche de [θ_Γ − θ_{Γ₀}] ramené à (−π, π].

Vérifications :
- angles petits : 2,2·10⁻¹⁵ ;
- amplitude 1, avec 9 880 chemins « sautés » : 1,6·10⁻¹⁴ ;
- **jacobien constant** : bloc σ₃σ₃ de J = Q à 5,5·10⁻¹⁴, même à grands angles ; blocs mixtes U(1)/chargés exactement nuls.

### 5.2 Loi conditionnelle exacte (action quelconque, U(1) compact)

    μ(dθ | φ̄) = Z⁻¹ Σ_{m} Σ_{r∈ℤⁿ} 1_{𝒟_m}(θ) e^{−S(θ)} 𝓗|_{Qθ = φ̄ + 2π(r − B_m)}(dθ),   B_m = Σ_Γ w_Γ m_Γ,   m_Γ = −n_Γ.

- La fibre est une réunion de morceaux de sous-espaces affines, indexés par la cellule de branches m et le relèvement r.
- Le facteur det(QQᵀ)^{−1/2} est constant et disparaît. Le mutant « sans jacobien » est donc **exact** en U(1).
- Les cellules sont séparées par des coupures de **codimension 1** (en SU(2), codimension 3).
  Une dynamique continue (route B) reste dans sa cellule, une marche avec sauts (route A) les traverse.

### 5.3 Action gaussienne non compacte (loi fermée)

On prend S = β θᵀMθ = (β/2) Σ_p θ_p², avec M = ½ rotᵀrot. C'est aussi chaque couleur de SU(2) en coordonnées ω à β → ∞ (§6).

Notations :
- Q⁺ = Qᵀ(QQᵀ)⁻¹ ;
- B est une base orthonormée de la tranche 𝒱 = ker Q ∩ (jauge résiduelle)^⊥, la jauge résiduelle étant dλ avec λ = 0 aux centres ;
- Γ = B(BᵀMB)⁻¹Bᵀ, c'est `picov.Classe.Gam` de S1.

Loi conditionnelle :

    θ = H φ̄ + v,   v ~ N(0, Γ/(2β)) sur 𝒱,   H = (1 − ΓM) Q⁺  (extension harmonique = solution KKT),
    S_min(φ̄) = β φ̄ᵀ M_c φ̄,   M_c = HᵀMH ;   F = rot θ ~ N(rot H φ̄, ½ rot Γ rotᵀ / β).

- La forme de Schur de Kimi, Σ₀ = β⁻¹(K⁻¹ − K⁻¹Qᵀ(QK⁻¹Qᵀ)⁻¹QK⁻¹) après fixation de jauge (K = 2M), est équivalente pour les
  observables invariantes.
- dim 𝒱 = 3(L⁴ − L_c⁴), donc ⟨S⟩ = S_min + 3(L⁴ − L_c⁴)/2 et la moyenne de β⟨θ_p²⟩ (à φ̄ = 0) vaut (1 − 3⁻⁴)/2 = **40/81**,
  indépendamment de L.
- Le tore ajoute 4 modes harmoniques plats (A constant). Q les voit (Q(a e_μ) = 3a), donc ils ne sont pas dans la fibre.

### 5.4 Route A gaussienne (loi fermée en κ = βε²)

    β·cov(θ) = (2M + QᵀQ/κ + P_G)⁻¹   (partie invariante ; P_G = projecteur sur la jauge résiduelle, sans effet sur les observables invariantes)

Contrôle : κ = 10⁻⁶ redonne la loi contrainte à 8·10⁻⁶ près, par deux calculs indépendants (FIB-U1-ROUTEA-LIMITE).

Table sur 6⁴, φ̄ = 0, valeurs exactes (contrainte dure : ⟨S⟩ = 1920, β Var θ_W = 1,834724757698) :

| κ = βε² | cas de FIB-AB | ⟨S⟩ (écart) | β Var(θ_W), boucle 3×3 des centres (écart) |
|---|---|---|---|
| 10⁻⁴ | — | 1920,084 (+0,004 %) | 1,836239 (+0,08 %) |
| 10⁻³ | — | 1920,800 (+0,042 %) | 1,849127 (+0,79 %) |
| 3·10⁻³ | — | 1922,167 (+0,113 %) | 1,873832 (+2,13 %) |
| 0,007 | β = 2,8, ε = 0,05 | 1924,276 (+0,223 %) | 1,912210 (+4,22 %) |
| 0,02 | β = 8, ε = 0,05 | 1928,404 (+0,438 %) | 1,988517 (+8,38 %) |
| 0,028 | β = 2,8, ε = 0,1 | 1929,986 (+0,520 %) | 2,018119 (+10,00 %) |
| 0,08 | β = 8, ε = 0,1 | 1934,964 (+0,779 %) | 2,111837 (+15,10 %) |
| 0,112 | β = 2,8, ε = 0,2 | 1936,381 (+0,853 %) | 2,138442 (+16,55 %) |
| 0,32 | β = 8, ε = 0,2 | 1939,698 (+1,026 %) | 2,200151 (+19,92 %) |
| 2,5 / 10 / 40 | β = 1000, ε = 0,05 / 0,1 / 0,2 | 1942,07 / 1942,39 / 1942,47 | 2,2437 / 2,2494 / 2,2509 (≈ loi libre) |

Lecture :
- Avec les ε du plan, la courbe n'est pas linéaire en ε².
  Par exemple, l'extrapolation linéaire de ⟨S⟩ à β = 8 depuis ε = 0,1 et 0,05 laisse +0,32 %, autant que l'effet du mutant (§6.3).
- À β = 1000, la route A avec ε ≥ 0,05 mesure presque la loi **libre**.
- La table complète (22 valeurs de κ) est dans `sorties/valeurs_u1.json`.

### 5.5 Valeurs gelées U(1), réseau 6⁴ (L_c = 2), φ̄ = 0

On classe les plaquettes par position (r = x mod 3) et par plan. Il reste 9 classes, selon deux critères :
- combien de côtés du plan franchissent un bord de bloc (coordonnée r = 2 dans le plan) ;
- combien de coordonnées transverses passent par le centre (r = 1).

Valeurs de β⟨θ_p²⟩ (multiplicités entre parenthèses) :

| côtés franchis \ transverses au centre | 2 | 1 | 0 |
|---|---|---|---|
| 2 (ex. plan 01, r = (2,2,·,·)) | 0,4673092809 (6) | 0,4747537091 (24) | 0,4804255256 (24) |
| 1 (ex. r = (0,2,·,·)) | 0,4905874009 (24) | 0,4927463133 (96) | 0,4943773844 (96) |
| 0 (ex. r = (0,0,·,·)) | 0,4984310324 (24) | 0,4987060909 (96) | 0,4989139643 (96) |

Autres valeurs :
- moyenne 40/81 = 0,493827160494 ; invariance par translation de 3 : 1,2·10⁻¹⁵ ;
- boucle 3×3 des centres (plan 01), β Var(θ_W) : **1,834724757698**, contre 2,251390404863 sans contrainte ;
- M_c[c,c] = **13,239314327096** ; spectre de M_c dans [0 ; 51,624007], avec 19 modes nuls : 15 de jauge grossière (L_c⁴ − 1)
  et 4 plats ;
- les 486 classes, la première ligne de M_c et la table κ complète sont dans `sorties/valeurs_u1.json`.

**Vecteur de 10 moments pour FIB-U1.** C'est la forme qu'attend le brouillon de GATES v4-T2 partie 1.

On pose m = (m₁, …, m₉ ; m₁₀) :
- m₁ à m₉ : β⟨θ_p²⟩ moyen des 9 types, dans l'ordre (côtés franchis, transverses au centre) = (2,2), (2,1), (2,0), (1,2),
  (1,1), (1,0), (0,2), (0,1), (0,0) ;
- m₁₀ : β Var(θ_W), pour la boucle 3×3 qui passe par les centres c(0), c(0) + 3e₀, c(0) + 3e₀ + 3e₁ et c(0) + 3e₁.

Propriétés :
- À φ̄ = 0, les 96 boucles de ce type ont la même valeur, et les 486 classes se réduisent exactement à ces 9 types
  (FIB-U1-TYPES : dispersion 3·10⁻¹⁶).
- Les moments physiques valent m/β, si bien qu'une seule table sert pour **tout** couplage, par exemple β ∈ {8 ; 2,8}.

| loi | m₁ | m₂ | m₃ | m₄ | m₅ | m₆ | m₇ | m₈ | m₉ | m₁₀ |
|---|---|---|---|---|---|---|---|---|---|---|
| exacte (contrainte dure) | 0,4673092809 | 0,4747537091 | 0,4804255256 | 0,4905874009 | 0,4927463133 | 0,4943773844 | 0,4984310324 | 0,4987060909 | 0,4989139643 | 1,8347247577 |
| route A, κ = 10⁻³ | 0,468910 | 0,475926 | 0,481286 | 0,490858 | 0,492937 | 0,494512 | 0,498460 | 0,498726 | 0,498927 | 1,849127 |
| route A, κ = 2·10⁻³ | 0,470314 | 0,476959 | 0,482049 | 0,491105 | 0,493112 | 0,494635 | 0,498487 | 0,498744 | 0,498940 | 1,862077 |
| route A, κ = 4·10⁻³ | 0,472675 | 0,478707 | 0,483346 | 0,491541 | 0,493423 | 0,494856 | 0,498535 | 0,498778 | 0,498963 | 1,884587 |
| route A, β = 8, ε = 0,05 (κ = 0,02) | 0,482297 | 0,485952 | 0,488807 | 0,493676 | 0,494971 | 0,495976 | 0,498785 | 0,498957 | 0,499091 | 1,988517 |
| route A, β = 8, ε = 0,1 (κ = 0,08) | 0,491382 | 0,492974 | 0,494239 | 0,496408 | 0,497033 | 0,497529 | 0,499142 | 0,499227 | 0,499294 | 2,111837 |
| route A, β = 8, ε = 0,2 (κ = 0,32) | 0,496781 | 0,497284 | 0,497689 | 0,498432 | 0,498639 | 0,498805 | 0,499433 | 0,499462 | 0,499484 | 2,200151 |
| route A, β = 2,8, ε = 0,05 (κ = 0,007) | 0,475435 | 0,480766 | 0,484884 | 0,492089 | 0,493816 | 0,495137 | 0,498597 | 0,498821 | 0,498994 | 1,912210 |
| route A, β = 2,8, ε = 0,1 (κ = 0,028) | 0,484681 | 0,487775 | 0,490203 | 0,494316 | 0,495445 | 0,496326 | 0,498865 | 0,499016 | 0,499134 | 2,018119 |
| route A, β = 2,8, ε = 0,2 (κ = 0,112) | 0,493088 | 0,494321 | 0,495304 | 0,497014 | 0,497506 | 0,497898 | 0,499227 | 0,499294 | 0,499347 | 2,138442 |

La loi exacte de la route A est donc connue **à chaque ε** : FIB-U1 peut comparer chaque ε, sans extrapolation (clé
`vecteur_m` de `valeurs_u1.json`, avec plus de chiffres).

**Mutant du brouillon v4-T2 pour FIB-U1** (V décalé de exp(0,05 iσ₃) sur le lien grossier (z = 0, μ = 0), références
inchangées) : décalage relatif exact des moments.
- m₁₀ : +4,02 % à β = 8, +1,41 % à β = 2,8.
- m₁ à m₉ : ≤ 0,11 %, car le décalage est dilué sur les 16 blocs.
- Le mutant se lit donc presque entièrement sur m₁₀. La boucle de m₁₀ doit bien être celle qui contient le lien décalé, ce qui
  est le cas avec la définition ci-dessus.

**Ancrage dans S1.** La table des plaquettes et M_c[c,c], calculées ici en espace direct, sont égales à celles de
`picov.Classe` de S1 (Bloch, KKT, Gam, moyenne sur les 16 moments de L_c = 2), à 5·10⁻¹⁵ et 4·10⁻¹³ (FIB-BINF-KKT-S1).
De plus, Q (énumération indépendante) = `op_Q` de `geom.py` à 3,6·10⁻¹⁵.

## 6. Limite β → ∞ (SU(2)) : la fibre linéarisée est le KKT de S1

### 6.1 Ordre dominant

- J(1) = Q ⊗ 1₃ (5,5·10⁻¹⁴ ; FIB-BINF-J1).
- L'espace tangent à la fibre en U = 1 est ker Q ⊗ ℝ³. Il contient la jauge résiduelle.
- Avec U = e^{iω·σ}, S = β Σ_a ω_aᵀMω_a + O(βω³) : chaque couleur est le cas gaussien du §5.3.

Pour V = e^{iȳ·σ} proche de 1, le fond est le minimiseur sous contrainte min{S : QA = ȳ} (KKT de S1, H = `picov.Classe.H` au
sens invariant). Les fluctuations de fibre ont pour covariance

    cov(ω) = Γ/(2β) ⊗ 1₃ ;   en coordonnées S1 (x = √2 ω) : Γ/β ;   en coordonnées WP5 (a = 2ω) : 2Γ/β.

À cet ordre, le facteur de coaire et la densité de Haar sont constants.

Valeurs gelées SU(2) à β → ∞, V = 1, sur 6⁴ (`sorties/valeurs_beta_infini.json`) :
- β⟨1 − P_p⟩ = **1,5 ×** la table du §5.5 ;
- moyenne **20/27 = 0,740740740741** = (3/4)(1 − 3⁻⁴), contre (3/4)(1 − 1/V) sans condition ;
- boucle 3×3 des centres : β⟨1 − ½Tr W⟩ = **2,752087136546**, contre 3,377085607294 sans contrainte.

### 6.2 Ordre du facteur de coaire

On développe log det(J Jᵀ)(e^{isa}) = 3 log det(QQᵀ) + c₁s + c₂s² + c₃s³ + …, avec a ~ N(0,1) par composante sur 6⁴.

**c₁ = 0.**
- Preuve : au premier ordre, toutes les corrections de J (rotations R(P*), L_g, E_g) sont des matrices de produit vectoriel.
  On a donc J = Q ⊗ 1 + s Σ C_i ⊗ K_i, avec K_i antisymétriques.
- D'où ∂ log det = 2 Σ tr((QQᵀ)⁻¹QC_iᵀ) · tr K_iᵀ = 0.
- Numérique : c₁ = 2,7·10⁻¹¹ (ajustement à 3 pas).

Ordres suivants :
- c₂ = 1837,5 ;
- **c₃ = −92,4 ≠ 0** : log det(J Jᵀ) n'est **pas** pair (désaccord D3, §7).

Le facteur de coaire modifie donc les observables à l'ordre relatif O(1/β).

### 6.3 Puissance du mutant « sans jacobien » (pour FIB-MUT)

Avec F = ½ log det J Jᵀ ≈ ¼ AᵀH_L A et cov(A) = Γ/(2β) ⊗ 1, on a, à l'ordre dominant,

    ⟨S⟩_correct − ⟨S⟩_mutant = −Cov(S, F) = −Tr((Γ ⊗ 1) H_L)/(8β),   Tr((Γ ⊗ 1) H_L) = 1336,0 ± 9,4

(Hutchinson, 16 vecteurs, 6⁴, `t8_puissance_mutant.py`). Rapporté à ⟨S⟩ = 5760, l'effet relatif sur ⟨1 − P⟩ vaut −0,029/β :

| β | 1000 | 16 | 8 | 2,8 |
|---|---|---|---|---|
| effet relatif | −0,003 % | −0,18 % | −0,36 % | ≈ −1,0 % (ordre dominant, peu fiable vers β ≈ 3) |

Le signe concorde avec le jouet : sans le facteur, la fibre est « plus chaude ».

**Taille statistique requise.** L'effet relatif est intensif (Tr et ⟨S⟩ sont tous deux extensifs). Par configuration
indépendante, σ(S)/⟨S⟩ = ⟨S⟩^{−1/2}, car en gaussien chaque mode a moyenne ½ et variance ½. On veut voir le mutant à 3σ sur la
**différence** des deux routes, soit 3 σ_rel √(2/N) ≤ effet :

| réseau | ⟨S⟩ (β → ∞) | σ_rel par configuration | N par route, β = 8 (effet 0,36 %) | N par route, β = 2,8 (effet ≈ 1,0 %) |
|---|---|---|---|---|
| 6⁴ | 5760 | 1,32 % | ≈ 240 | ≈ 30 |
| 12⁴ | 92 160 | 0,33 % | ≈ 15 | ≈ 2 |

Ces N sont des configurations **décorrélées**. La colonne β = 2,8 repose sur l'effet de l'ordre dominant, à confirmer par WP12b.
**Le mutant se voit à moindre coût sur 12⁴**. Une tolérance de puissance σ_comb ≤ 0,5 % ne suffit pas à β = 2,8 : un effet de 1 %
donne alors |z| ≈ 2. Il faut σ_comb ≲ 0,3 %.

### 6.4 Domaine aux β de FIB-AB

Rapports RG-DOMAINE de WP6 (grandL L = 12) : à β = 2,3, 44 % des logarithmes de boucle ont θ > π/2, 0,12 % ont θ > 15π/16,
et θ max = 3,138 ; à β = 16, la fraction au-delà de π/2 tombe à 3,8·10⁻⁴.

Près de W_Γ = −1, L_g diverge comme π/(π − θ) : le potentiel ½ log det J Jᵀ et la projection de Newton deviennent raides.
- En SU(2), la coupure est de codimension 3, la fibre reste connexe et la mesure y est supprimée (det(J Jᵀ)^{−1/2} → 0).
- Mais la route B à β = 2,8 a besoin d'un contrôle de réversibilité (Lelièvre et al. 2019) et d'un pas adapté. La route A y est
  insensible.

## 7. Deux dérivations indépendantes

**Protocole.**
- Kimi K3.1 Pro par `kimi_max`, en bac à sable systemd (PrivateTmp : mes fichiers de travail, dans le tmp privé, lui étaient invisibles).
- Prompt `kimi/prompt_aveugle.txt`, sha256 72eb95d316e17d6a… : énoncé et conventions, questions Q1-Q6, **aucune réponse**.
- Durée environ 25 min.
- Audit de la trace : aucune lecture hors de son dossier.
- Fichiers dans `validation/fibre_coaire/kimi/` : réponse, scripts et sorties de Kimi.

| point | Opus | Kimi | arbitre | verdict |
|---|---|---|---|---|
| facteur de coaire, normalisation (2π²)^{n−N} | e^{−S} det(JJᵀ)^{−1/2} d𝓗 | identique | jouet SymPy (§4) | **accord** |
| invariance de trivialisation et de jauge | oui | oui | FIB-TRIVIALISATION, FIB-BINF-JAUGE | **accord** |
| L_g, L_d, E_g | §3.1 | forme en X, X², algébriquement identique | DF 60 chiffres | **accord** |
| jacobien général des chemins | §3.3 | identique (B₀ = 1 − Σ w B) | DF : 10⁻²⁸ (jouets), 10⁻¹¹ (carte réelle) | **accord** |
| jouet : J, det JJᵀ, fibre, W ~ Haar | §4 | identique | SymPy | **accord** |
| jouet : I₂/I₁, mutant, écarts relatifs (β = 2, 8, 1000) | §4 | identiques à 12 chiffres | Bessel et quadrature | **accord** |
| U(1) : jacobien constant, branches, loi gaussienne, pénalité en βε² | §5 | identique (Kimi ajoute les relèvements r ∈ ℤⁿ, repris au §5.2) | t5 | **accord** |
| β → ∞ : J(1) = Q ⊗ 1, ker Q, KKT, 20/27 | §6 | identique ; sans condition 3/4 − 3/(4L⁴) | t6 | **accord** |
| **D1** potentiel de RATTLE | S + ½ log det(JM⁻¹Jᵀ) | S + ½ log det M − ½ log det(JM⁻¹Jᵀ), et « rien à ajouter pour M = 1 » (contradictoire) | SymPy (Liouville) et 6 runs RATTLE | **Opus** |
| **D2** E_g en θ → π | nnᵀ | nnᵀ − π⁻¹[n×] | limite SymPy | **Opus** (point mineur) |
| **D3** pas de terme linéaire dans log det | trace de couleur | parité par inversion, Ū(U⁻¹) = Ū(U)⁻¹ | c₃ = −92,4 ; \|Ū(U⁻¹) − Ū(U)⁻¹\| = 0,46 | conclusion identique ; **argument de Kimi faux** |

### Détail des trois désaccords

**D1 (le seul à conséquence pratique).**
- Erreur de Kimi : il prend comme mesure invariante la mesure « naïve » δ(ξ)δ(JM⁻¹p) dq dp au lieu de la mesure de Liouville
  de T*F. Il manque le facteur det(J M⁻¹ Jᵀ).
- Arbitrage par `t7_rattle_arbitrage.py` :
  - SymPy : ∫e^{−p²/2g}dp = √(2πg), donc la marginale vaut e^{−V}σ_M, et g/A = m₁m₂ exactement ;
  - RATTLE et Metropolis, avec projection de Newton et contrôle de réversibilité, sur la courbe y = 1,2 sin x de ℝ²,
    pour M ∈ {1, diag(1, 4)} et trois potentiels.
- Résultats :
  - les 6 lois échantillonnées suivent la prédiction de Liouville (|z| ≤ 1,4) ;
  - la prédiction de Kimi est rejetée (|z| = 278 à 426) ;
  - seul le potentiel « Opus » retrouve la cible : ⟨cos x⟩ = −0,2829288041, ⟨cos 2x⟩ = 0,2054655606, à |z| ≤ 0,9.

**D2.** sin θ cos θ/θ et sin²θ/θ² tendent vers 0 en π (DLOG-EG-PI-SYMPY).

**D3.** L'inversion lien par lien n'inverse pas les holonomies (l'ordre des produits est conservé). La parité est donc fausse en
non abélien. Seul le terme linéaire s'annule (FIB-BINF-PAS-PAIR).

**Verdict FIB-DERIV.**
- Les deux dérivations s'accordent sur le jacobien de la coaire et sur toutes les valeurs exactes.
- Le désaccord de mise en œuvre (D1) et les deux points mineurs sont tranchés mécaniquement.
- SymPy : jacobien analytique = différences finies sur le jouet à 3,6·10⁻²⁸, sous la gate de 10⁻⁸.
- U(1) : la loi conditionnelle exacte est écrite (§5.2, §5.3).
- **PASS.**

## 8. Cas tests gelés pour WP12a et WP12b

Toutes les valeurs ci-dessous sont dans `sorties/valeurs_*.json` (hachées dans `sorties/SHA256SUMS`).

| id | objet | valeurs | usage proposé |
|---|---|---|---|
| CT-1 | jouet SU(2), loi correcte | ⟨x⟩ = I₂/I₁ et ⟨x²⟩ (table du §4), **indépendantes de V et de t** | routes A et B sur le jouet ; à lancer avec 2 V et 2 t |
| CT-2 | jouet SU(2), mutant | table du §4 (β = 2 : 0,1544 contre 0,4331) | FIB-MUT, démonstration de puissance |
| CT-3 | jouet U(1) | I₁/I₀ ; mutant identique | contrainte U(1) |
| CT-4 | U(1) gaussien 6⁴, contrainte dure | vecteur m de 10 moments (§5.5) ; ⟨S⟩ = 1920 ; M_c[c,c] = 13,239314327096 | FIB-U1 (mode gaussien non compact, tout β : moments = m/β) |
| CT-5 | U(1) gaussien 6⁴, pénalité | m(κ) aux 6 ε de FIB-AB et aux 3 κ proposés (§5.5) ; table en κ de ⟨S⟩ et de m₁₀ (§5.4) | route A, à **chaque** ε et pas seulement extrapolée |
| CT-6 | SU(2), β → ∞, 6⁴ | 1,5 × CT-4 ; 20/27 ; boucle 2,752087136546 | FIB-GAUSS (référence, en β⟨·⟩) |
| CT-7 | moteur RATTLE sur ℝ² (courbe y = 1,2 sin x, V = ½cos x + y²/2) | cible ⟨cos x⟩ = −0,2829288041, ⟨cos 2x⟩ = 0,2054655606 ; lois des variantes A et C (§7) | test unitaire du potentiel de la route B, M = 1 et M ≠ 1 |
| CT-8 | jacobien de la vraie carte | fixtures 9⁴ de WP6 : J = DF ≤ 10⁻⁸ ; implémentation de référence `balaban_np.Bloc.jac` | jacobien analytique de la route B |
| CT-9 | invariance de jauge | log det JJᵀ inchangé (≤ 10⁻¹²) sous jauge de Haar | instrument de la route B |
| CT-10 | puissance du mutant | effet −0,029/β sur ⟨1 − P⟩ (6⁴) | dimensionnement de la statistique de FIB-AB et FIB-MUT |

## 9. Gates

**FIB-DERIV (catalogue v1, ce lot) : PASS.**
- Lignes de `sorties/verdicts.txt` (run final n°2, 26/10 vers 0 h 28) : **57 PASS, 0 FAIL, 0 SKIP**, 8 scripts à rc 0.
  Le run n°1 (56 PASS) précédait l'ajout de FIB-U1-TYPES ; les autres lignes sont identiques entre les deux runs, sauf le dernier
  chiffre d'un écart (M_c, 3,5 contre 4,0·10⁻¹³, ordre de sommation BLAS).
- Parmi elles : FIB-DERIV-JOUET-DF, FIB-DERIV-CHEMINS-B/C et FIB-DERIV-BALABAN-DF (SU(2), différences finies), FIB-JOUET-U1-J (U(1),
  différences finies, 6·10⁻¹⁰ ≤ 10⁻⁸) et FIB-U1-CARTE-*/JACOBIEN-CONSTANT (U(1) sur 6⁴), FIB-COAIRE-JOUET-SYMPY/NUM,
  FIB-U1-* (loi exacte), FIB-ARB-* (arbitrage D1), DLOG-EG-PI-SYMPY (D2), FIB-BINF-PAS-PAIR (D3).
- Seuils écrits avant les runs. Deux seuils de **mes** scripts ont été corrigés après un premier passage, et c'est consigné :
  - FIB-BINF-J1 : 10⁻¹⁴ → 10⁻¹². Premier passage à 5,5·10⁻¹⁴, arrondi de sommes de 10 473 termes en F64.
  - FIB-BINF-SANS-LINEAIRE : l'extraction de c₁ à 2 pas était polluée par c₅ (c₁ apparent 1,4·10⁻⁴). Elle se fait maintenant à
    3 pas, seuil 10⁻⁶ inchangé.
- Aucune gate du catalogue n'est touchée.

**Propositions pour le pilote de la tranche 2** (non gelées, à geler en v4 avant les runs de WP12) :
1. **FIB-JOUET** (nouvelle, instrument WP12a). Les routes A (ε → 0) et B reproduisent CT-1 à 3σ, à β ∈ {2 ; 8}, pour t ∈ {½ ; 0,3}
   et deux V. Mutant de la route B : CT-2 doit sortir ROUGE (β = 2 : 0,154 contre 0,433).
2. **FIB-U1 précisée.** Mode U(1) gaussien non compact sur 6⁴, φ̄ = 0, avec le vecteur m de 10 moments (§5.5, CT-4) :
   la route B contre m/β, la route A à chaque ε contre m(κ)/β (CT-5).
   Le mutant « V décalé de 0,05 » du brouillon est assez puissant sur m₁₀ (+4,0 % à β = 8).
   L'U(1) compact n'est pas adapté à la route B (cellules de branches, §5.2).
3. **FIB-AB : choix de ε.** ε = (κ/β)^{1/2}, κ ∈ {10⁻³ ; 2·10⁻³ ; 4·10⁻³} (β = 8 : ε = 0,011 ; 0,016 ; 0,022 ; β = 2,8 : 0,019 ;
   0,027 ; 0,038), extrapolation linéaire en κ. Cela remplace ε ∈ {0,2 ; 0,1 ; 0,05}, dont le biais est non linéaire (CT-5).
4. **FIB-MUT.** À juger sur FIB-JOUET (puissance garantie) et sur FIB-AB à β = 2,8, de préférence sur la cellule **12⁴**, avec une
   statistique dimensionnée par CT-10 (§6.3 : σ_comb ≲ 0,3 % sur ⟨1 − P⟩).
   **Pas** sur FIB-GAUSS : à β = 1000, l'effet vaut 3·10⁻⁵ en relatif contre une tolérance de 3 %, donc ECHEC_OUTIL par construction.
5. **FIB-GAUSS.** Référence CT-6. Les corrections en O(1/β) valent environ 0,1 % à β = 1000 : la tolérance de 3 % est lâche,
   et 0,5 % serait tenable si la statistique le permet.
6. **FIB-RATTLE-POT** (nouvelle). CT-7 sur le moteur RATTLE de WP12a (M = 1 et M = diag(1, 4)) : la variante « Opus » retrouve la
   cible à 3σ, et les variantes sans terme ou avec signe opposé sortent ROUGES.

**Note sur le brouillon `scratch/pilote-t2/GATES_v4T2_partie1_projet.md`** (lu le 26/10 vers 0 h 25, non modifié) :
- FIB-AB y garde ε ∈ {0,2 ; 0,1 ; 0,05} avec un ajustement O₀ + cε².
  - D'après CT-5, le biais n'est pas linéaire en ε² à ces valeurs : β = 8, ⟨S⟩ +1,03 / +0,78 / +0,44 %, m₁₀ +20 / +15 / +8 %.
  - L'extrapolation de ⟨S⟩ à partir de ε = 0,1 et 0,05 laisse +0,32 %, autant que l'effet du mutant à β = 8.
  - Un ajustement à 3 points peut avoir un χ² acceptable et rester biaisé.
  - Recommandation : la proposition 3 ci-dessus, ou au minimum la valeur à ε de la loi gaussienne exacte comme témoin de forme.
- FIB-MUT y demande max|z| ≥ 3,144 à β = 2,8 avec σ_comb ≤ 0,5 %. Pour un effet attendu ≈ 1 % (ordre dominant, 6⁴), cela donne
  |z| ≈ 2, donc un risque d'ECHEC_OUTIL. Il faut σ_comb ≲ 0,3 % (cellule 12⁴) et FIB-JOUET en démonstration de puissance.
  L'alternative « ou FIB-GAUSS ROUGE » ne peut pas se produire (effet 3·10⁻⁵).
- FIB-U1 : le vecteur m, la loi à chaque ε et la puissance du mutant sont livrés (§5.5).
- FIB-JAC et FIB-LOGDET : la forme fermée (§3.3) et les dérivées de L_g et E_g (§3.2) sont livrées. La référence numérique est
  `balaban_np.Bloc.jac`.

## 10. Reproduction

    bash crates/kvu/validation/fibre_coaire/lancer.sh [dossier_sorties]

- Environ 10 min sur acemagic (run n°2, machine chargée), CPU seul, calc_borne 10G, mémoire de pointe de l'ordre de 3 Go.
- Python système, en `-I -B`, fils BLAS bornés à 8. Versions : `sorties/versions.txt` (python 3.14.4, sympy 1.14.0, mpmath 1.3.0,
  numpy 2.5.1).
- Lectures seules : `crates/kvu/validation/rg/fixtures` (WP6, sha256 contrôlés contre le manifeste) et le dossier S1 du chantier
  vision (`geom.py`, `picov.py`, `bundle_q3`) ; rien n'y est écrit.

| script | objet | durée |
|---|---|---|
| `t1_dlog.py` | §3.1-3.2, D2 | 3 s |
| `t2_jouet.py` | §4, CT-1 à CT-3 | 7 s |
| `t3_chemins.py` | §3.3 (jouets B et C) | 4 s |
| `t4_balaban.py` | carte réelle, CT-8 | 1,9 min |
| `t5_u1.py` | §5, CT-4, CT-5, vecteur m | 1,4 min |
| `t6_beta_infini.py` | §6, S1, CT-6, CT-9, D3 | 46 s |
| `t7_rattle_arbitrage.py` | D1, CT-7 | 3,4 min |
| `t8_puissance_mutant.py` | §6.3, CT-10 | 2 min |

## 11. Limites (dites telles quelles)

- L'énoncé porte sur la fibre **sans** restriction de petits champs (χ₀ de Bałaban). Les ensembles de mesure nulle (coupure,
  défaut de rang) sont écartés et non traités.
- Seules les observables invariantes de jauge sont couvertes. La fixation de jauge de type Faddeev–Popov pour la Hessienne en
  tranche (WP13) n'est pas dérivée ici.
- La puissance du mutant n'est connue qu'à l'ordre dominant en 1/β, sur 6⁴. À β = 2,8 elle n'est qu'un ordre de grandeur.
  Le mutant doit être mesuré directement par WP12b.
- Ni les corrections O(1/β) aux covariances de FIB-GAUSS, ni Sp(4) et G₂ (tranche 3) ne sont traitées.
