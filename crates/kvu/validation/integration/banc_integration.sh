#!/usr/bin/env bash
# banc_integration.sh — bancs de l'intégration (unifie/integ) sur la machine COURANTE.
#
#   bash banc_integration.sh [--sortie DIR] [--bin KEVINOTRON]
#
# Ce que l'intégration demande (en plus de BANC_UNIFIE.md de T9, qui couvrait L = 8, 16, 32, 48) :
#   su2    L = 16, 24, 32, 48 × F64, F32 × lots 1, 16 ; référence su2_hip (acemagic) / su2_cuda (gamer) lancée dans
#          la MÊME fenêtre (F64, une chaîne) : « ancien contre nouveau » ; gate de non-régression ≥ 0,9 ;
#   cond   L = 16, 24, 32, modes rapide et ref_rng, sur le GPU puis sur le CPU (moteur Rust rayon) ;
#   fibre, obs : L = 16, 24, 32.
# Chaque banc prend le verrou GPU de la machine par fenêtres de 600 s au plus et le rend entre deux fenêtres
# (cli/bench.rs). Seuls de petits JSON sont écrits : DIR/<banc>_<machine>.json.
set -u
ICI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${KVU_BIN:-}"
SORTIE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --sortie) SORTIE="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "option inconnue : $1" >&2; exit 2 ;;
  esac
done
if grep -q "AI MAX+ 395" /proc/cpuinfo 2>/dev/null; then
  MACHINE=acemagic; SEUIL=110; PRE="nice -n 10"
  : "${BIN:=/chemin/vers/projet/chantiers/kevinotron_unification/bin/wp1/acemagic/kvu}"
elif grep -q "i5-14600KF" /proc/cpuinfo 2>/dev/null; then
  MACHINE=gamer; SEUIL=50; PRE="nice -n 5 taskset -c 12-19"
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
banc su2 su2 --L 16,24,32,48 --precision f64,f32 --chaines 1,16 --budget-s 600
banc cond cond --L 16,24,32 --budget-s 600
banc cond_cpu cond --L 16,24,32 --backend cpu --budget-s 600
banc fibre fibre --L 16,24,32 --budget-s 300
banc obs obs --L 16,24,32 --budget-s 300
exit $rc
