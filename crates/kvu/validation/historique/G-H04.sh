#!/usr/bin/env bash
# G-H04 (ancienne P1) : bain thermique exact, 7 alpha, F64, backend de la machine (HIP sur acemagic, CUDA sur gamer).
GATE=G-H04
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
exiger_gpu; binaire_ok "$GATE"
sous_verrou $PRE_CPU python3 "$HIST/gh04_bain.py" "$BIN" --precision f64 2>"$SCRATCH/gh04.err" | tee "$SCRATCH/gh04.out"
cat "$SCRATCH/gh04.err" >&2
grep -q '^\(PASS\|FAIL\) G-H04' "$SCRATCH/gh04.out" || fail "$GATE" "aucune ligne de verdict (voir stderr)"
exit 0
