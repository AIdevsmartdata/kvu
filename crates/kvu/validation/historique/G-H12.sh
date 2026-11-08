#!/usr/bin/env bash
# G-H12 (ancienne P5) : KS par permutation des plaquettes locales, β = 2,8, L = 6 et 8, CUDA unifié contre les
# références AMD (gpu_port/ref_amd/ens) : p > 0,01 (B = 2000, graine 27092026).
# Tourne SUR ACEMAGIC, sur les dumps du gamer rapatriés par `kevinotron fetch` (sha256 vérifiés).
GATE=G-H12
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
reserve_a acemagic
[ "${KVU_SANS_GPU:-0}" = 1 ] && exit 0
binaire_ok "$GATE"
exiger_ssd "$GATE" 1
rapatrier_gh06 || { fail "$GATE" "ensemble gh06 du gamer absent ou incomplet (fetch gamer:$STORE_GAMER_V/gh06)"; exit 0; }
nice -n 10 "$PY" "$HIST/analyse_historique.py" p5 --cuda "$STORE_V/gh06_gamer" --json "$SCRATCH/gh12.json"
exit 0
