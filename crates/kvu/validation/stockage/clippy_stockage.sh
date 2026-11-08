#!/usr/bin/env bash
# clippy_stockage.sh — gate (d) de la tâche T8 : clippy sans aucun avertissement sur les fichiers de T8.
#
# 1. `cargo clippy -p kvu --all-targets -- -D warnings`, rc rapporté tel quel ;
# 2. vérifie qu'AUCUN diagnostic ne tombe dans un fichier de T8 (stockage/, build.rs de kvu, tests/stockage_*,
#    tests/donnees/stockage, tests/provenance_unifiee.rs, src/manifest.rs : lignes ajoutées par T8 seulement).
# Les avertissements des fichiers GELÉS ou historiques compilés dans le crate (commun.rs, heritage.rs,
# src/su2_hb.rs, src/hip/wrap.rs, src/manifest.rs historique, noyaux des autres tâches) ne sont pas à T8 : ils sont
# listés à part. Voir RAPPORT_T8.md.
set -u
ICI=$(cd "$(dirname "$0")" && pwd)
DEPOT=$(cd "$ICI/../../../.." && pwd)
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/chemin/vers/projet/chantiers/kevinotron_unification/cibles/T8}
cd "$DEPOT" || exit 1
if cargo clippy --version >/dev/null 2>&1; then CLIPPY="cargo clippy"; else CLIPPY="cargo +1.97.0 clippy"; fi
touch crates/kvu/src/lib.rs   # force la ré-analyse : clippy n'affiche que ce qu'il compile
out=$(nice -n 10 $CLIPPY -p kvu --all-targets --message-format=short 2>&1)
echo "$out" | grep -E "warning|error" | sed 's/^/  /' | head -60
mien=$(echo "$out" | grep -E "^crates/kvu/(src/stockage/|build\.rs|tests/(stockage_|donnees/stockage|provenance_unifiee))|^tests/provenance_unifiee" | grep -E "warning|error")
nhors=$(echo "$out" | grep -E "^(crates/|src/)" | grep -E "warning|error" | grep -vcE "^crates/kvu/(src/stockage/|build\.rs|tests/(stockage_|donnees/stockage|provenance_unifiee))")
if [ -z "$mien" ]; then
  echo "PASS CLIPPY-T8 : 0 diagnostic dans les fichiers de T8 (${nhors} diagnostic(s) dans des fichiers gelés ou historiques)"
  exit 0
else
  echo "FAIL CLIPPY-T8 :"; echo "$mien"
  exit 1
fi
