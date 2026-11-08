# `kvu cond facteur3` — fibre dure de DEC (facteur 3) et noyau de réponse de la moyenne conditionnelle (test T3-D)

Module `crates/kvu/src/cond3/`. Spécification, oracles et règle de décision : `chantiers/T3_moyenne_conditionnelle_2026-11/{SPEC_T3,ORACLES_T3,PREREG_T3}.md` (v1.2).
Extension du bain conditionnel `kvu cond` (facteur 2, paires U₁U₂ = V) aux TRIPLETS U₁U₂U₃ = V.

## Objet
Mesure de Wilson SU(2) (ou U(1) gaussien non compact) sachant le champ de bloc DEC(U)(X, μ) = U(3X, μ)U(3X + e_μ, μ)U(3X + 2e_μ, μ) = V figé :
loi dure sur la fibre, `U₃ = U₂†U₁†V` asservi, les autres liens étant libres. Mesure, par paire (plaquette observée p, sonde ē), le noyau de réponse
R_p(ē) = ∂E[½ Tr U_p | V]/∂V_ē par l'identité de relèvement exacte `R̂ = E[X̄(O)] − Cov(O, σ̄)` (trois relèvements exacts à divergence nulle, moyenne σ̄),
puis Ŝ = R̂^A · R̂^B (deux répliques indépendantes) sans biais pour |R|².

## Conventions
- Quaternions q = q₀ + i q·σ, produit (ab)₀ = a₀b₀ − a·b, (ab)_v = a₀b_v + b₀a_v − a_v × b_v ; lien li = μ·V + s (s indice C, x₀ le plus lent) ; plaquette pi = plan·V + s ;
  lien grossier ci = μ·Vc + Sc ; ligne = (3X + j e_μ, μ), j = 0, 1, 2.
- Agrafe K_ℓ : Σ_{p∋ℓ} ½ Tr U_p = ½ Tr(U_ℓ K_ℓ†) ; ligne : K_eff(U₁) = K₁ + V K₃† U₂†, K_eff(U₂) = K₂ + U₁† V K₃†.
- Scores : σ¹ = β(U₁K₁†)_v, σ² = β R(U₁)(U₂K₂†)_v, σ³ = β R(U₁U₂)(U₃K₃†)_v ; identité de Ward grossière exacte (vérifiée à 1e-13 à chaque contrôle).
- Un pas = 1 balayage de bain thermique + n_OR = 1 de surrelaxation, sous-étapes (pour μ = 0..d) : libres de parité paire, impaire, tous les U₁ des lignes, tous les U₂ ; Philox à compteur
  (lien, balayage, tirage, COND) hérité de `kvu cond` (tirage de x₀ exact : `cond::cpu::lien_rapide`).

## Commandes
- `kvu cond facteur3 repliques` : cellule complète (départ : `--cfg` NPY SU(2) d = 4 ou `--wilson` ; V = DEC figé ; répliques A et B ; sommes globales FP64 Σ O, Σ σ, Σ O σ par bloc de temps ;
  `--reduire` : sommes par tranche de Ŝ et histogrammes de charge de tube au lieu des accumulateurs complets ; `--backend hip|cuda|cpu|auto`).
- `kvu cond facteur3 effectifs` : effectifs de la géométrie et des paires par tranche (oracle O-0).
- `kvu cond facteur3 tube-classe | tube-f | tube-hist` : tubes de charge (oracle G0.8, passe 1 des classes de charge).

## Interface de groupe
`cond3::modele::Modele` : un seul code de géométrie (`geom`, d ∈ {2, 4}), d'ordonnancement, de mesure (`mesure`) et d'estimation (`reduction`) pour SU(2) (`su2`) et le U(1) gaussien (`u1`) :
les oracles 2D (O-3) et U(1) (O-4) passent donc par le MÊME code que la production. GPU : `cond3::gpu` + `noyaux/cond3.cu` (SU(2), d = 4, HIP et CUDA, FP64 sans FMA).

## Validation
`crates/kvu/tests/cond3_base.rs` (agrafes, Ward, contrainte, covariance de jauge, 23 permutations d'axes, relèvements exacts, canaux de Stein contre différences finies, mutants) et tests unitaires de
`geom`, `paires`, `tube` (table des tubes égale à la force brute sur TOUTES les classes de L = 6), `mesure` (τ_int d'un AR(1) connu, réduction de production = somme naïve par paire, lots de tailles différentes).
`crates/kvu/tests/cond3_gpu.rs` (`#[ignore]`, à lancer en fenêtre GPU : concordance CPU/GPU, rejeu au bit). Les oracles statistiques sont dans `chantiers/T3_moyenne_conditionnelle_2026-11/implementation/oracles/`.

## Limites connues (documentées par la relecture)
- Table des tubes : indices stockés en `u16` (plan·V_c + site) : L_c ⩽ 8, soit L ⩽ 24 ; au-delà, `TableTubes::nouvelle` s'arrête avec un message (pas de débordement silencieux).
- Numéro de balayage Philox : `u32` ; `sweep0 + n_dec + n_mes` ne doit pas le dépasser (refusé par `verifier_compteur_philox`).
- `StatsEchantillonneur` : `majs`/`essais` couvrent tous les balayages au CPU mais seulement les balayages de contrôle au GPU ; `plafond` est cumulé sur tous les balayages.
- GPU : SU(2), d = 4, fenêtre b ⩾ 2, scores moyens, ordonnancement simultané, sans canaux d'oracle (les oracles passent par le CPU).
- Canaux d'oracle (`--canaux FILE.json`) : indices non validés (hors bornes = panique) ; usage réservé aux oracles. Ordonnancement séquentiel : réservé à l'oracle O-2b, sans mutants u3/dérive (refusés).
- Renormalisation des liens : pas de garde contre une norme nulle (un quaternion exactement nul n'apparaît pas en fonctionnement normal ; G-CONTRAINTE, G-TRAJ et le défaut d'unitarité le détecteraient).
