#!/usr/bin/env bash
# Tests de CODE (pas de production) — β = 2,5 uniquement (hors grille de production) ; on n'inspecte que
# RC, manifestes, identité bit à bit et chronométrage, JAMAIS les différences entre bords.
set -u
cd "$(dirname "$0")"
bash build.sh || exit 10
B=./bin/su2_bord; REF=$HOME/ym4_calcul/su2_cuda/bin/su2_cuda
mkdir -p tests; nf=0
run() { echo "+ $*" >> tests/cmds.log; "$@" >> tests/stdout.log 2>> tests/stderr.log; local r=$?; echo "rc=$r :: $*" >> tests/rc.log; [ $r = 0 ] || nf=$((nf+1)); }
exec 9>/tmp/gamer_gpu.lock; flock -w 7200 9 || { echo "verrou GPU non obtenu"; exit 11; }
# C0 : chemin périodique bit-identique à su2_cuda
run $REF --L 8 --beta 2.5 --seed 28092800001 --start hot --therm 50 --nmeas 200 --skip 1 --nor 4 --outdir tests/c0_ref
run $B   --L 8 --beta 2.5 --seed 28092800001 --start hot --therm 50 --nmeas 200 --skip 1 --nor 4 --outdir tests/c0_bord
for f in therm.csv measurements.csv; do cmp tests/c0_ref/$f tests/c0_bord/$f && echo "C0 $f IDENTIQUE" >> tests/rc.log || { echo "C0 $f DIFFERENT" >> tests/rc.log; nf=$((nf+1)); }; done
# C1 / C6 : gel intact, déterminisme, jauge exacte (β = 2,5, L = 8)
for bc in cold hot; do run $B --bord --L 8 --beta 2.5 --seed 28092800002 --bc $bc --bc-seed 28092800003 --pre 200 --therm 200 --nmeas 1000 --bloc 100 --nor 4 --outdir tests/t_$bc; done
run $B --bord --L 8 --beta 2.5 --seed 28092800002 --bc eq --bc-seed 28092800004 --pre 200 --therm 200 --nmeas 1000 --bloc 100 --nor 4 --outdir tests/t_eq
run $B --bord --L 8 --beta 2.5 --seed 28092800002 --bc eq --bc-seed 28092800004 --jauge 28092800005 --pre 200 --therm 200 --nmeas 1000 --bloc 100 --nor 4 --outdir tests/t_eqg
run $B --bord --L 8 --beta 2.5 --seed 28092800002 --bc cold --bc-seed 28092800003 --pre 200 --therm 200 --nmeas 1000 --bloc 100 --nor 4 --outdir tests/t_cold_bis
cmp tests/t_cold/blocs.csv tests/t_cold_bis/blocs.csv && echo "C6 determinisme IDENTIQUE" >> tests/rc.log || { echo "C6 DIFFERENT" >> tests/rc.log; nf=$((nf+1)); }
# chronométrage L = 12, 16
for L in 12 16; do run $B --bord --L $L --beta 2.5 --seed 28092800006 --bc hot --bc-seed 28092800007 --pre 200 --therm 0 --nmeas 2000 --bloc 100 --nor 4 --outdir tests/chrono_L$L; done
run $B --bord --L 8 --beta 2.5 --seed 28092800006 --bc hot --bc-seed 28092800007 --pre 200 --therm 0 --nmeas 2000 --bloc 100 --nor 4 --outdir tests/chrono_L8
echo "echecs=$nf" >> tests/rc.log
exit $nf
