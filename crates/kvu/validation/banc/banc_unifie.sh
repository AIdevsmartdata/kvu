#!/usr/bin/env bash
# banc_unifie.sh — banc du kevinotron unifié SUR LA MACHINE COURANTE (ARCHITECTURE_UNIFIEE.md §6.7).
#
#   bash banc_unifie.sh [--sortie DIR] [--budget-s 600] [--seulement su2|cond|fibre|obs]...
#
# Quatre bancs, chacun sous le verrou GPU de la machine par fenêtres de 600 s au plus (le GPU est rendu entre
# deux fenêtres, voir cli/bench.rs) :
#   su2    L = 8,16,32,48 × F64,F32 × lots de 1,16, avec en référence su2_hip (acemagic) ou su2_cuda (gamer)
#          lancé dans la MÊME fenêtre ; gate de non-régression : F64 unifié >= 0,9 × référence ;
#   cond   L = 16,24,32, modes rapide et ref_rng ;  fibre  L = 16,24,32 ;  obs  L = 16,24,32.
# Sortie : DIR/<banc>_<machine>.json (JSON de `kevinotron bench`). Le rapport (BANC_UNIFIE.md) est produit par
# rapport_banc.py à partir des JSON des deux machines.
# Garde SSD : seuls de petits JSON sont écrits ; le script refuse sous 110 Go (acemagic) / 50 Go (gamer).
set -u
ICI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${KVU_BIN:-kevinotron}"
BUDGET=600
SORTIE=""
SEULEMENT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --sortie) SORTIE="$2"; shift 2;;
    --budget-s) BUDGET="$2"; shift 2;;
    --seulement) SEULEMENT="$SEULEMENT $2"; shift 2;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "option inconnue : $1" >&2; exit 2;;
  esac
done
MACHINE="${KVU_MACHINE:-}"
if [ -z "$MACHINE" ]; then
  if grep -q "AI MAX+ 395" /proc/cpuinfo 2>/dev/null; then MACHINE=acemagic
  elif grep -q "i5-14600KF" /proc/cpuinfo 2>/dev/null; then MACHINE=gamer
  else MACHINE=inconnue; fi
fi
[ -z "$SORTIE" ] && SORTIE="$ICI/resultats/$MACHINE"
case "$MACHINE" in gamer) SEUIL=50; PRE="nice -n 5 taskset -c 12-19";; *) SEUIL=110; PRE="nice -n 10";; esac
D="$SORTIE"; while [ ! -e "$D" ] && [ "$D" != / ]; do D="$(dirname "$D")"; done   # premier parent existant
LIBRE="$(df --output=avail -BG "$D" 2>/dev/null | tail -1 | tr -dc '0-9')"
if [ -z "$LIBRE" ] || [ "$LIBRE" -lt "$SEUIL" ]; then
  echo "REFUS garde SSD : ${LIBRE:-?} Go libres, seuil $SEUIL Go" >&2; exit 7
fi
mkdir -p "$SORTIE" || exit 4

veut() { [ -z "$SEULEMENT" ] && return 0; case " $SEULEMENT " in *" $1 "*) return 0;; esac; return 1; }
rc_global=0
lancer() { # famille, arguments…
  local fam="$1"; shift
  veut "$fam" || return 0
  echo "== banc $fam sur $MACHINE (budget ${BUDGET} s)" >&2
  $PRE "$BIN" bench "$fam" "$@" --budget-s "$BUDGET" --json --sortie "$SORTIE/${fam}_${MACHINE}.json" >/dev/null
  local rc=$?
  echo "== banc $fam : rc=$rc" >&2
  [ "$rc" -ne 0 ] && rc_global=$rc
  return 0
}
lancer su2   --L 8,16,32,48 --precision f64,f32 --chaines 1,16
lancer cond  --L 16,24,32
lancer fibre --L 16,24,32
lancer obs   --L 16,24,32
echo "résultats : $SORTIE (rc global $rc_global ; 8 = gate de non-régression rouge ou cas en échec)" >&2
exit "$rc_global"
