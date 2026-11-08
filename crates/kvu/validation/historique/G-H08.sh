#!/usr/bin/env bash
# G-H08 (ancienne P2) : départs chauds HIP et CUDA de mêmes graines : max|Δ| < 1e-13 sur la configuration initiale.
# Les deux machines produisent L = 6 et 8 (β = 2,8, graines 27092900006 et 27092900008, 20 pas, un dump par pas,
# départ inclus) ; le gamer ne rend aucun verdict (RECHERCHE), acemagic rapatrie ses dumps (`fetch`) et compare.
GATE=G-H08
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
exiger_gpu; binaire_ok "$GATE"
exiger_ssd "$GATE" 1
OUT="$STORE_V/gh08/$MACHINE"; rm -rf "$OUT"; mkdir -p "$OUT"
for L in 6 8; do
  sous_verrou $PRE_CPU "$BIN" su2 gen --L "$L" --beta 2.8 --graine "2709290000$L" --start hot --therm 0 --nmeas 20 --skip 1 --nor 4 \
    --dump-every 1 --dump-initial --precision f64 --options compat-prod --outdir "$OUT/L$L" >"$SCRATCH/gh08_L$L.log" 2>&1 \
    || { [ "$MACHINE" = acemagic ] && fail "$GATE" "run L=$L en échec : $(tail -1 "$SCRATCH/gh08_L$L.log")"; exit 0; }
done
reserve_a acemagic
rm -rf "$STORE_V/gh08/gamer"; mkdir -p "$STORE_V/gh08"
"$BIN" fetch "gamer:$STORE_GAMER_V/gh08/gamer" "$STORE_V/gh08/gamer" >"$SCRATCH/gh08_fetch.log" 2>&1 \
  || { fail "$GATE" "fetch des dumps du gamer en échec : $(tail -1 "$SCRATCH/gh08_fetch.log")"; exit 0; }
"$PY" "$HIST/analyse_historique.py" p2 --p2-hip "$OUT" --p2-cuda "$STORE_V/gh08/gamer"
exit 0
