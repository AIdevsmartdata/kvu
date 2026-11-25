# Fragment de documentation : `kvu rg apparier` et `kvu rg choisir-c` (candidat c02b)

MCRG de Swendsen à deux réseaux (`rg apparier`) et choix de la constante d'agrafe c par dispersion P/R/C minimale (`rg choisir-c`).
Estimateurs préenregistrés dans `chantiers/mcrg_desert_2026-11-10/` (PREREG, AMENDEMENT_1 et 2) : inchangés par ce candidat.

## Ce que le candidat c02b ajoute (relecture GPT-6.1 Sol du candidat c02)
- Bissection : une racine exactement à une borne de la fenêtre est rendue (elle était perdue) ; l'égalité exacte au dernier point de grille du pilote est traitée.
- Une variance bootstrap nulle ou non finie (P, R, C ou leurs différences) est REFUSÉE (message « ajustement non identifiable »), sauf tension nulle légitime (solutions P/R/C identiques à chaque réplicat) ; tout `null` issu d'un NaN/infini est refusé avant l'écriture du JSON (rc ≠ 0).
- Bornes validées avant toute mesure : `--b` ∈ {1, 2}, `--c` fini, `--pas` ∈ [1, 16], `--degre` ∈ [1, 8], `--fenetre` finie ≥ 0, `--bootstrap-b` ∈ [20, 10^6], `--bootstrap-bloc` ∈ [1, 10^6], `--fils` ∈ [1, 4096] (0 refusé).
- `--fils` borne aussi le bootstrap parallèle (il tournait dans le pool global). Les JSON ne dépendent pas du nombre de fils.
- `choisir-c` exige que les entrées soient des `kvu-rg-apparier/1` décrivant le même problème : tous les paramètres (volumes, b, nombres de blocages, degré, fenêtre, bootstrap, graine) identiques sauf `c`, mêmes ensembles (nombres de chaînes et de configurations du fin et de chaque point de grille, mêmes β′ et L).
- `--sortie DIR` : DIR est créé s'il est neuf. Provenance : `apparier.json.manifest.json` / `choix_c.json.manifest.json` à côté du JSON (binaire, source_hash, git_head, machine, paramètres, entrées avec leurs sha256 pour `choisir-c`), écrits APRÈS le JSON, dont le contenu est inchangé à l'octet.

## Limites connues, non corrigées (estimateurs préenregistrés)
- O7 : l'erreur interpolée de `choisir-c` somme les erreurs des Δβ(c) en quadrature, sans covariance. Sensibilité sur les barres publiées, SOUS l'hypothèse ρ = 1 entre les Δβ(c) (non mesurée ; avec des poids négatifs ρ = 1 ne maximise pas l'erreur, la borne générale est (Σ|w_i|σ_i)²) : ≤ 3 % si ρ = 1, ≤ 9 % pour toute corrélation (borne Σ|w_i|σ_i), jamais mesuré ; le terme dominant est pente × 0,025.
- O9 : le bootstrap est conditionné aux réplicats qui ont une racine dans la fenêtre (seuil : 20 valides). Aucune censure pour β_haut ≥ 2,5 dans la production existante ; jusqu'à 77 % de réplicats éliminés à β_haut ≤ 2,0.
Détails chiffrés, démonstrations et corrections proposées : `chantiers/kvu_modules_2026-11-11/ORCHESTRE/p4fix/NOTE_ESTIMATEURS.md`.
