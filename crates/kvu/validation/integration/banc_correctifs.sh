#!/usr/bin/env bash
# banc_correctifs.sh — bancs rejoués après les correctifs du vérificateur (04/10), sur la machine COURANTE.
#
#   bash banc_correctifs.sh [--sortie DIR] [--bin KEVINOTRON] [--cb]
#
#   su2   L = 16, 24, 32, 48 × F64, F32, une chaîne : noyaux (1 HB + 4 OR) ET temps mur de `su2 gen` (champ `mur`),
#         référence su2_hip / su2_cuda dans la même fenêtre (gate de non-régression F64) ;
#   cond  L = 16, 24, 32, rapide et ref_rng, GPU puis CPU (moteur Rust), champ de départ THERMALISÉ (300 pas HB + 4 OR,
#         β = 2,3) ; sur acemagic, en plus, les configurations de production de noyau2 (L = 16, 24, lues en place) ;
#   --cb  (acemagic) cb.py ref_rng et rapide aux mêmes L, numba 1 et 4 fils (noyau2/bench.py, configurations de
#         production ; L = 32 = pavage 2⁴ de L = 16, comme le README de noyau2).
# Chaque banc GPU prend le verrou de sa machine par fenêtres de 600 s au plus et le rend entre deux fenêtres
# (cli/bench.rs). Seuls de petits JSON sont écrits : DIR/<banc>_<machine>.json.
set -u
ICI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${KVU_BIN:-}"
SORTIE=""
CB=0
while [ $# -gt 0 ]; do
  case "$1" in
    --sortie) SORTIE="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --cb) CB=1; shift ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "option inconnue : $1" >&2; exit 2 ;;
  esac
done
if grep -q "AI MAX+ 395" /proc/cpuinfo 2>/dev/null; then
  MACHINE=acemagic; SEUIL=110; PRE="nice -n 10"
  : "${BIN:=/chemin/vers/projet/chantiers/kevinotron_unification/bin/wp1/acemagic/kvu}"
elif grep -q "i5-14600KF" /proc/cpuinfo 2>/dev/null; then
  MACHINE=gamer; SEUIL=60; PRE="nice -n 5"   # l'affinité 12-19 est posée par le binaire lui-même
  : "${BIN:=$HOME/kvu_unif/bin/wp1/kvu}"
else
  echo "machine inconnue" >&2; exit 2
fi
: "${SORTIE:=$ICI/resultats/$MACHINE}"
mkdir -p "$SORTIE"
LIBRE="$(df --output=avail -BG / | tail -1 | tr -dc '0-9')"
if [ "${LIBRE:-0}" -lt "$SEUIL" ]; then echo "REFUS garde SSD : $LIBRE Go libres, seuil $SEUIL Go" >&2; exit 7; fi

rc=0
banc() { # nom args…
  local nom="$1"; shift
  echo "== [$MACHINE] bench $* ($(date +%T))" >&2
  # shellcheck disable=SC2086
  $PRE "$BIN" bench "$@" --json --sortie "$SORTIE/${nom}_$MACHINE.json" >/dev/null || { echo "bench $nom : rc=$?" >&2; rc=1; }
}
banc su2_mur su2 --L 16,24,32,48 --precision f64,f32 --chaines 1 --budget-s 900
banc cond_therm cond --L 16,24,32 --budget-s 600
banc cond_therm_cpu cond --L 16,24,32 --backend cpu --budget-s 900
NOYAU2=/chemin/vers/projet/outils/ym4_noyaux/noyau2
if [ "$MACHINE" = acemagic ] && [ -d "$NOYAU2/data" ]; then
  for L in 16 24; do
    banc "cond_prod_L$L" cond --L "$L" --cfg "$NOYAU2/data/L${L}_b2.3.npy" --budget-s 300
    banc "cond_prod_L${L}_cpu" cond --L "$L" --cfg "$NOYAU2/data/L${L}_b2.3.npy" --backend cpu --budget-s 300
  done
fi
if [ "$CB" = 1 ] && [ "$MACHINE" = acemagic ] && [ -x /chemin/vers/projet/outils/ym4_noyaux/venv/bin/python ]; then
  for fils in 1 4; do
    echo "== [$MACHINE] cb.py numba $fils fil(s) ($(date +%T))" >&2
    NUMBA_NUM_THREADS=$fils REF_SWEEP_MAX=0 REF_MES_MAX=0 \
      NUMBA_CACHE_DIR="$HOME/.cache/kevinotron/numba_noyau2" \
      $PRE /chemin/vers/projet/outils/ym4_noyaux/venv/bin/python -B "$NOYAU2/bench.py" "$SORTIE/cbpy_${fils}fil_$MACHINE.json" 16 24 32 \
      >/dev/null || { echo "cb.py $fils fil(s) : rc=$?" >&2; rc=1; }
  done
fi
exit $rc
