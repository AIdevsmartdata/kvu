#!/usr/bin/env bash
# Balayage en β par lots : 16 valeurs de β, UNE chaîne par β, UN seul lancement (aux petits L le GPU est limité par la
# latence de lancement : un lot amortit les ~40 lancements par pas sur toutes les chaînes).
#
# `--beta` donne une liste (une valeur par chaîne), `--graine S --graine-pas 1` fait graine_i = S + i. La chaîne i du lot
# donne les mêmes octets que la même chaîne lancée seule (LOT-1), donc un balayage peut être refait point par point.
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
BETAS="2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5"
if [ "$EXEMPLE_MODE" = petit ]; then
  L=4; THERM=20; NMEAS=20; BACKEND="--backend cpu"; PREC=f64   # le CPU ne calcule qu en f64
else
  L=8; THERM=2000; NMEAS=20000; BACKEND=""; PREC=f32
fi
VERROU="${VERROU:-/tmp/kevinotron_gpu.lock}"
lancer "$KVU_BIN" su2 gen --L "$L" --beta "$BETAS" --graine 20261004000 --graine-pas 1 --start hot \
  --therm "$THERM" --nmeas "$NMEAS" --skip 1 --nor 4 --precision "$PREC" $BACKEND --outdir "$SORTIE/balayage_L$L" || exit $?
echo "Sorties : $SORTIE/balayage_L$L/c000 … c015 (measurements.csv : plaquette et Polyakov par pas)"
