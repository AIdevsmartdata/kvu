# Licences des dépendances

kvu est sous licence Apache-2.0. Aucune dépendance n'est copiée dans ce dépôt : cargo télécharge les crates ci-dessous
depuis crates.io, aux versions exactes de `Cargo.lock`. Toutes sont sous licences permissives compatibles avec
Apache-2.0 ; aucune n'est sous GPL, LGPL, AGPL ou licence non commerciale. Relevé du 8 novembre 2026 (champ `license` du
`Cargo.toml` de chaque crate, à la version de `Cargo.lock`).

## Crates Rust (38 dépendances, directes et transitives)

| licence | crates |
|---|---|
| MIT OR Apache-2.0 | anstream, anstyle, anstyle-parse, anstyle-query, anstyle-wincon, cfg-if, clap, clap_builder, clap_derive, clap_lex, colorchoice, crossbeam-deque, crossbeam-epoch, crossbeam-utils, cudarc (fonctionnalité `cuda` seulement), either, heck, is_terminal_polyfill, itoa, once_cell_polyfill, proc-macro2, quote, rayon, rayon-core, serde, serde_core, serde_derive, serde_json, syn (2.0 et 3.0), windows-link, windows-sys |
| Apache-2.0 OR MIT | utf8parse |
| (MIT OR Apache-2.0) AND Unicode-3.0 | unicode-ident |
| MIT | strsim, zmij |
| Unlicense OR MIT | memchr |
| ISC | libloading |

Dépendances directes (`crates/kvu/Cargo.toml`) : rayon, libloading, clap, serde, serde_json, et cudarc (facultative).

## Code ou données d'origine tierce dans le dépôt

- `crates/kvu/src/cond/numpy_tables.rs` : constantes du ziggurat de NumPy (BSD à 3 clauses ; texte de la licence dans le
  fichier). Voir `NOTICE`.
- Le générateur Philox4x32-10 est réimplémenté d'après sa publication (Salmon et al., SC '11) ; aucun code de Random123.

## À l'exécution (non distribué, non lié à la compilation)

- GPU : ROCm (HIP, hiprtc) d'AMD et pilote/NVRTC de NVIDIA, chargés dynamiquement s'ils sont présents.
- `zstd` (binaire système, licence BSD ou GPLv2 au choix de son auteur) : appelé comme programme externe, jamais lié.
- Python 3 et numpy, scipy (BSD) pour les scripts de référence appelés par certains tests ; d'autres scripts de
  `crates/kvu/validation/` emploient aussi mpmath, sympy (BSD) ou numba (BSD à 2 clauses).
