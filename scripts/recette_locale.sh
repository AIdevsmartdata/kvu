#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# recette_locale.sh : recette de kvu sur UNE machine, depuis la racine du dépôt (ou d'une archive `git archive`).
#
#   1. compilation `--release --locked` du binaire ;
#   2. suite complète `cargo test --release --locked --no-fail-fast` (les tests GPU sont `#[ignore]` : ils ne tournent pas) ;
#   3. liste des tests selon cargo (`-- --list`, `-- --list --ignored`) et contrôle, si une référence existe, qu'aucun test
#      de la référence n'a été retiré ni désactivé et que chaque test actif a été exécuté (scripts/recette/noms_tests.py) ;
#   4. clippy `--all-targets -D warnings` ;
#   5. compilation seule (`--no-run`) des tests avec la fonctionnalité `cuda` : vérifie que le backend CUDA compile, sans GPU.
#
# Usage : scripts/recette_locale.sh [--offline] [--sans-clippy] [--sans-cuda] [--reference FICHIER] [DOSSIER_SORTIE]
#   DOSSIER_SORTIE : journaux (défaut : target/recette_locale). Variables : CARGO_BUILD_JOBS (défaut : 8).
#   Référence par défaut : scripts/recette/noms_tests/v0.1.0.cpu.txt (suite sans la fonctionnalité `cuda`).
# Code de retour : 0 si tout est vert, 1 sinon. Le résumé est écrit sur la sortie standard et dans DOSSIER_SORTIE/RESUME.txt.
set -u
OFF=(); CLIPPY=1; CUDA=1; REF=""; SORTIE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --offline) OFF=(--offline) ;;
    --sans-clippy) CLIPPY=0 ;;
    --sans-cuda) CUDA=0 ;;
    --reference) REF=${2:?--reference demande un fichier}; shift ;;
    -h|--help) sed -n '3,16p' "$0"; exit 0 ;;
    -*) echo "option inconnue : $1" >&2; exit 2 ;;
    *) SORTIE=$1 ;;
  esac
  shift
done
RACINE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$RACINE" || exit 2
SORTIE=${SORTIE:-$RACINE/target/recette_locale}
mkdir -p "$SORTIE" || exit 2
SORTIE=$(cd "$SORTIE" && pwd)
REF=${REF:-$RACINE/scripts/recette/noms_tests/v0.1.0.cpu.txt}
JOBS=${CARGO_BUILD_JOBS:-8}
export CARGO_TERM_COLOR=never
R="$SORTIE/RESUME.txt"; : > "$R"
dire() { echo "$*" | tee -a "$R"; }
etape() { local nom=$1; shift; local t0 rc; t0=$(date +%s); "$@" > "$SORTIE/$nom.log" 2>&1; rc=$?; echo "$rc $(( $(date +%s) - t0 ))"; }
VERT=1

dire "RECETTE kvu $(git -C "$RACINE" rev-parse --short HEAD 2>/dev/null || echo archive) ; $(cargo --version) ; $(date -Is)"
read -r rc d < <(etape build cargo build --release --locked "${OFF[@]}" -j "$JOBS" -p kvu)
[ "$rc" = 0 ] && dire "PASS build ($d s)" || { dire "FAIL build rc=$rc (voir $SORTIE/build.log)"; VERT=0; }

read -r rc d < <(etape tests cargo test --release --locked "${OFF[@]}" --no-fail-fast -j "$JOBS" -p kvu)
read -r p f i n < <(awk '/^test result:/ {p+=$4; f+=$6; i+=$8; n++} END {print p+0, f+0, i+0, n+0}' "$SORTIE/tests.log")
# $p additionne les lignes « test result » : il compte aussi les tests qui relancent leur propre exécutable dans un processus
# enfant (tests/su2_cpu.rs). Le décompte exact des tests est celui de la liste de cargo (étape suivante).
if [ "$rc" = 0 ] && [ "$f" = 0 ] && [ "$p" -gt 0 ]; then dire "PASS tests : rc 0, 0 échec ($p « passed » et $i « ignored » additionnés sur $n lignes « test result » ; $d s, compilation comprise)"
else dire "FAIL tests rc=$rc : $p passés, $f échecs, $i ignorés (voir $SORTIE/tests.log)"; grep -E "^test .* FAILED$" "$SORTIE/tests.log" | head -20 | tee -a "$R"; VERT=0; fi

read -r rc1 _ < <(etape liste cargo test --release --locked "${OFF[@]}" -j "$JOBS" -p kvu -- --list)
read -r rc2 _ < <(etape ignores cargo test --release --locked "${OFF[@]}" -j "$JOBS" -p kvu -- --list --ignored)
if [ "$rc1" != 0 ] || [ "$rc2" != 0 ]; then dire "FAIL liste des tests (rc $rc1 / $rc2)"; VERT=0
elif [ -f "$REF" ]; then
  if python3 -I "$RACINE/scripts/recette/noms_tests.py" verifier "$REF" "$SORTIE/liste.log" "$SORTIE/ignores.log" "$SORTIE/tests.log" > "$SORTIE/noms.log" 2>&1
  then dire "PASS noms des tests : $(tr '\n' ' ' < "$SORTIE/noms.log")"; else dire "FAIL noms des tests : $(cat "$SORTIE/noms.log")"; VERT=0; fi
else
  python3 -I "$RACINE/scripts/recette/noms_tests.py" reference "$SORTIE/liste.log" "$SORTIE/ignores.log" --commit inconnu --etape cpu --source recette_locale > /dev/null 2> "$SORTIE/noms.log"
  dire "INFO aucune référence de noms ($REF) : contrôle des noms non fait ; $(tail -1 "$SORTIE/noms.log")"
fi

if [ "$CLIPPY" = 1 ]; then
  read -r rc d < <(etape clippy cargo clippy --locked "${OFF[@]}" -j "$JOBS" -p kvu --all-targets -- -D warnings)
  [ "$rc" = 0 ] && dire "PASS clippy ($d s)" || { dire "FAIL clippy rc=$rc (voir $SORTIE/clippy.log)"; VERT=0; }
fi
if [ "$CUDA" = 1 ]; then
  read -r rc d < <(etape cuda cargo test --release --locked "${OFF[@]}" -j "$JOBS" -p kvu --features cuda --no-run)
  [ "$rc" = 0 ] && dire "PASS compilation cuda --no-run ($d s)" || { dire "FAIL compilation cuda rc=$rc (voir $SORTIE/cuda.log)"; VERT=0; }
fi
[ "$VERT" = 1 ] && { dire "VERDICT VERT"; exit 0; } || { dire "VERDICT ROUGE"; exit 1; }
