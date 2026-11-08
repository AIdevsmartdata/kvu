#!/usr/bin/env bash
# G-H10 (ancienne P4, CUDA) : critères C1-C4 de PREREG_CRITERES sur les 18 chaînes CUDA unifiées (stationnarité,
# chaud contre froid, chaudes entre elles, espacement). Seuils gelés : |z| < 3 ; 1000 >= 5 tau_int.
# Tourne SUR ACEMAGIC, sur les dumps du gamer rapatriés par `kevinotron fetch` (sha256 vérifiés).
GATE=G-H10
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
reserve_a acemagic
[ "${KVU_SANS_GPU:-0}" = 1 ] && exit 0
binaire_ok "$GATE"
exiger_ssd "$GATE" 1
rapatrier_gh06 || { fail "$GATE" "ensemble gh06 du gamer absent ou incomplet (fetch gamer:$STORE_GAMER_V/gh06)"; exit 0; }
nice -n 10 "$PY" "$HIST/analyse_historique.py" p4cuda --cuda "$STORE_V/gh06_gamer" --json "$SCRATCH/gh10.json"
exit 0
