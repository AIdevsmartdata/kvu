#!/usr/bin/env bash
# G-H09 (ancienne P3) : plaquette CUDA unifié contre les références AMD, 6 cellules : |z| < 3 et chi2 p > 0,01.
# Tourne SUR ACEMAGIC, sur les dumps du gamer rapatriés par `kevinotron fetch` (sha256 vérifiés).
GATE=G-H09
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
reserve_a acemagic
[ "${KVU_SANS_GPU:-0}" = 1 ] && exit 0
binaire_ok "$GATE"
exiger_ssd "$GATE" 1
rapatrier_gh06 || { fail "$GATE" "ensemble gh06 du gamer absent ou incomplet (fetch gamer:$STORE_GAMER_V/gh06)"; exit 0; }
nice -n 10 "$PY" "$HIST/analyse_historique.py" p3 --cuda "$STORE_V/gh06_gamer" --json "$SCRATCH/gh09.json"
exit 0
