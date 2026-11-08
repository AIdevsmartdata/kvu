#!/usr/bin/env bash
# G-H13 (ancienne P6) : intégrité des dumps unifiés, RECALCULÉE SUR ACEMAGIC : forme (4,L,L,L,L,2,2) complex128,
# |UU† − 1| ≤ 1e-13, |det − 1| ≤ 1e-13, plaquette numpy = CSV ≤ 1e-12. Dumps du gamer (rapatriés) et de acemagic.
GATE=G-H13
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
reserve_a acemagic
[ "${KVU_SANS_GPU:-0}" = 1 ] && exit 0
binaire_ok "$GATE"
exiger_ssd "$GATE" 1
rapatrier_gh06 || { fail "$GATE" "ensemble gh06 du gamer absent ou incomplet"; exit 0; }
[ "$(nb_complete "$STORE_V/gh07")" -ge 18 ] || { fail "$GATE" "ensemble gh07 absent ou incomplet (G-H07 doit passer avant)"; exit 0; }
nice -n 10 "$PY" "$HIST/analyse_historique.py" p6 --cuda "$STORE_V/gh06_gamer" --hip "$STORE_V/gh07" --json "$SCRATCH/gh13.json"
exit 0
