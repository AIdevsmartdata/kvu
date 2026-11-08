#!/usr/bin/env bash
# gates_obs.sh — gates GPU des observables (OBS-1 à OBS-4), tâche T7.
#
# À lancer SOUS LE VERROU DE LA MACHINE (le script ne le prend pas : il est tenu par l'appelant) :
#   acemagic : flock -w 7200 /tmp/kevinotron_gpu.lock bash crates/kvu/validation/obs/gates_obs.sh
#   gamer    : flock -w 7200 /tmp/gamer_gpu.lock bash crates/kvu/validation/obs/gates_obs.sh --cuda
#
# Options : --cuda        compile avec la feature cuda (gamer)
#           --sans-banc   saute le banc rapporté (obs_banc_rapporte)
# Variables : CARGO_TARGET_DIR (défaut : <racine>/target), KVU_OBS_LOG (défaut : $CARGO_TARGET_DIR/gates_obs.log)
#
# Sortie : une ligne `PASS|FAIL|SKIP|REPORT|INFO <gate>` par gate, puis le code de sortie :
#   0 = OBS-1 (critère absolu 1e-15) et son mutant OBS1-MUTANT, OBS-2, OBS-3, OBS-4-L16 (et OBS-ENLIGNE) PASS ;
#   8 = au moins un FAIL ou une gate absente/SKIP ; 3 = pas de GPU (toutes SKIP).
#   OBS-4-L8 est INFO (arbitrage D3, GATES v1) : rapportée, elle ne change pas le code.
set -u
RACINE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$RACINE" || exit 4
FEATURES=()
BANC=1
for a in "$@"; do
  case "$a" in
    --cuda) FEATURES=(--features cuda) ;;
    --sans-banc) BANC=0 ;;
    *) echo "option inconnue : $a" >&2; exit 2 ;;
  esac
done
: "${CARGO_TARGET_DIR:=$RACINE/target}"
export CARGO_TARGET_DIR
LOG="${KVU_OBS_LOG:-$CARGO_TARGET_DIR/gates_obs.log}"
# CUDA (gamer) : bibliothèques NVRTC 13.3 ; cœurs E obligatoires.
if [ "${#FEATURES[@]}" -gt 0 ]; then
  export LD_LIBRARY_PATH="/usr/local/cuda-13.3/lib64:${LD_LIBRARY_PATH:-}"
  PREFIXE=(nice -n 5 taskset -c 12-19)
else
  PREFIXE=(nice -n 10)
fi

df --output=avail -BG / | tail -1 | sed 's/^/SSD libre : /'
FILTRES=(obs_1_ obs_2_ obs_3_ obs_4_ obs_enligne)
[ "$BANC" = 1 ] && FILTRES+=(obs_banc)
: > "$LOG"
for f in "${FILTRES[@]}"; do
  "${PREFIXE[@]}" cargo test --release -p kvu "${FEATURES[@]}" --test obs_gpu --offline -- \
      --ignored --nocapture --test-threads=1 "$f" >> "$LOG" 2>&1
  echo "[rc=$? filtre=$f]" >> "$LOG"
done
grep -E '(PASS|FAIL|SKIP|REPORT|BANC) |INFO OBS-4-L8 |^GPU |^Q_top|panicked|\[rc=' "$LOG"

ok=1
for g in OBS-1 OBS1-MUTANT OBS-2 OBS-3 OBS-4-L16 OBS-ENLIGNE; do
  if grep -q "PASS $g" "$LOG"; then :; else echo "ROUGE : $g n'a pas rendu PASS"; ok=0; fi
done
if grep -q "SKIP OBS" "$LOG" && ! grep -q "PASS OBS-1" "$LOG"; then
  echo "SKIP : pas de GPU utilisable (ne vaut pas vert)"
  exit 3
fi
if grep -q "panicked" "$LOG"; then ok=0; fi
[ "$ok" = 1 ] && { echo "RC=0 : OBS-1, OBS1-MUTANT, OBS-2, OBS-3, OBS-4-L16 PASS ; OBS-4-L8 (INFO) rapporté ci-dessus"; exit 0; }
echo "RC=8 : gates OBS rouges (voir $LOG)"
exit 8
