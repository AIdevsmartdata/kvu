# Fragment v0.2 : module spectro (`kvu su2 spectro run`), EXPÉRIMENTAL

Spectroscopie de torelons SU(2) sur boîte rectangulaire, CPU F64, sans torsion (générateur `su2::tube`). Opérateurs : lignes de Polyakov
enroulées le long de `--enroule`, lissage APE spatial (`--niveaux`, `--alpha`), impulsions transverses (`--moments na:nb`), temps = `--temps`.
Sorties par configuration : `corr.csv` (corrélateur symétrisé C_{ss'}(τ), t-moyenné), `phi.csv` (⟨φ_s⟩, jamais soustrait), `plaq.csv`, `ponc.csv`
(option `--ponctuel`), `manifeste.json`, `etat.bin` (reprise `--reprendre`). Analyse : `scripts/spectro/analyse_spectro.py` (jackknife par blocs, GEVP,
énergie effective en cosh, plateau). Portes : `su2::spectro::tests` (invariance de jauge, pure jauge, centre, accord au bit avec `obs polyakov_corr`
sur un cube, oracle synthétique, APE contre matrices 2×2, mutants). Statut : relu par une seule famille ; critères et résultats dans
`chantiers/deux_voies_2026-11-16/kvu_spectro/`.
