#!/usr/bin/env bash
# RT-1, RT-2 (et RT-1b, RT-auto) : runtime GPU unifié de T2 (tests/gpu_runtime.rs, tests/gpu_cache.rs, #[ignore = "GPU"]).
# RT-1 : noyaux d'essai (remplissage, saxpy FP64, réduction à arbre fixe) corrects, réduction identique à l'octet à la
# référence CPU ; RT-2 : arguments du compilateur (golden), CUBIN/HSACO chargé, cache (miss, hit, entrée corrompue évincée).
# Tourne dans la fenêtre de groupe GPU (verrou tenu par l'orchestrateur) ; prend lui-même le verrou sinon.
# Sortie : les lignes `PASS|FAIL|SKIP RT-… : détail` des tests.
GATE=RT
. "$(dirname "${BASH_SOURCE[0]}")/commun.sh"
exiger_gpu
exiger_ssd "$GATE" 8
FEAT=""
if [ "$MACHINE" = gamer ]; then
  FEAT="--features cuda"; export PATH="$HOME/.cargo/bin:$PATH" LD_LIBRARY_PATH="/usr/local/cuda-13.3/lib64:${LD_LIBRARY_PATH:-}"
fi
LOG="$SCRATCH/rt.log"
( cd "$RACINE" && sous_verrou $PRE_CPU cargo test --release -p kvu $FEAT --test gpu_runtime --test gpu_cache -- --ignored --nocapture --test-threads=1 >"$LOG" 2>&1 )
rc=$?
grep -E '^(PASS|FAIL|SKIP) ' "$LOG"
if [ "$rc" -ne 0 ] && ! grep -q '^FAIL ' "$LOG"; then
  echo "FAIL RT-1 cargo test rc=$rc : $(grep -E 'error|panicked' "$LOG" | head -2 | tr '\n' ' ')"
fi
exit 0
