#!/usr/bin/env bash
# Ensembles du « désert » à grand L (48-64) sur acemagic : F32, dumps q64, mesures en ligne, projection SSD.
#
# Pourquoi : à L = 64 une configuration c16 pèse 4,3 Go, q64 2,1 Go ; acemagic n'a que ~160 Go libres (seuil de la garde
# SSD : 100 Go), soit ~60 Go utiles. D'où : F32 (champ 1,07 Go, pas ≈ 0,32 s au lieu de 1,2 s en F64), `--format q64`,
# peu de dumps, et des MESURES EN LIGNE (Wilson, flot) qui évitent de stocker les configurations.
#
# 1. `store du --projeter` dit AVANT le départ si la garde SSD accepte le run (rc 7 sinon) ;
# 2. `su2 gen` refuse de lui-même un run dont la projection dépasse la marge (rien n'est écrit) ;
# 3. le gamer est ≈ 2,5× plus rapide en F32 ; on rapatrie par `kevinotron fetch gamer:…` et on revérifie sur acemagic.
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
if [ "$EXEMPLE_MODE" = petit ]; then
  L=8; PAS=40; DUMP=20; MESURES="wilson:R2xT2:10,flot:0.01:0.05:20"; BACKEND="--backend cpu"; THERM=10; PREC=f64   # le CPU ne calcule qu en f64
else
  L=64; PAS=10000; DUMP=1000; MESURES="wilson:R8xT8:10,flot:0.01:3:20"; BACKEND=""; THERM=1000; PREC=f32
fi
NDUMPS=$((PAS / DUMP + 1))
echo "# Projection : L=$L, q64 = $(taille_go "$L" q64) Go par configuration, $NDUMPS dumps = $(awk -v a="$(taille_go "$L" q64)" -v n="$NDUMPS" 'BEGIN{printf "%.1f", a*n}') Go (c16 : $(taille_go "$L" c16) Go chacune)"
echo "# Durées estimées (§2) : pas ≈ 0,32 s (acemagic F32) / 0,13 s (gamer F32) à L = 64 → $PAS pas ≈ $(awk -v n="$PAS" 'BEGIN{printf "%.0f", n*0.32/60}') min (acemagic), $(awk -v n="$PAS" 'BEGIN{printf "%.0f", n*0.13/60}') min (gamer)"
lancer "$KVU_BIN" store du --projeter "$L:1:$NDUMPS:q64:$((PAS / 500 + 1))" || exit $?
VERROU="${VERROU:-/tmp/kevinotron_gpu.lock}"
lancer "$KVU_BIN" su2 gen --L "$L" --beta 2.5 --graine 20261004 --start hot --therm "$THERM" --nmeas "$PAS" --skip 1 --nor 4 \
  --precision "$PREC" --format q64 --mesures "$MESURES" --dump-every "$DUMP" --reprise-every 500 $BACKEND \
  --outdir "$SORTIE/desert_L$L" || exit $?
echo "# Un run interrompu se REPREND à l'octet :  $KVU_BIN su2 gen --reprendre $SORTIE/desert_L$L"
