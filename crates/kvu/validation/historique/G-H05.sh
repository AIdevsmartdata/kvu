#!/usr/bin/env bash
# G-H05 (ancienne P7) : reproductibilité à l'octet sur chaque machine (= G-DET). Deux runs identiques
# (L = 6, β = 2,8, therm 30, mesures 30, un dump tous les 10) donnent les mêmes octets (CSV et dumps) ;
# graine + 1 donne autre chose.
GATE=G-H05
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
exiger_gpu; binaire_ok "$GATE"
D="$SCRATCH/gh05"; rm -rf "$D"; mkdir -p "$D"
run() { sous_verrou $PRE_CPU "$BIN" su2 gen --L 6 --beta 2.8 --graine "$1" --start hot --therm 30 --nmeas 30 --skip 1 --nor 4 \
          --dump-every 10 --precision f64 --options compat-prod --outdir "$D/$2" >"$D/$2.log" 2>&1; }
run 27092806011 a && run 27092806011 b && run 27092806012 c || { fail "$GATE" "un des runs a échoué : $(tail -2 "$D"/*.log | tr '\n' ' ')"; exit 0; }
if ! diff -rq -x manifest.json -x COMPLETE -x config_run.json "$D/a" "$D/b" >"$D/diff_ab.txt"; then
  fail "$GATE" "deux runs identiques diffèrent : $(head -3 "$D/diff_ab.txt" | tr '\n' ' ')"; exit 0
fi
if diff -rq -x manifest.json -x COMPLETE -x config_run.json "$D/a" "$D/c" >/dev/null; then
  fail "$GATE" "graine + 1 donne les mêmes octets (le test n'a aucune puissance)"; exit 0
fi
nf="$(find "$D/a" -type f ! -name manifest.json ! -name COMPLETE | wc -l)"
pass "$GATE" "double run identique à l'octet ($nf fichiers), graine + 1 différente"
exit 0
