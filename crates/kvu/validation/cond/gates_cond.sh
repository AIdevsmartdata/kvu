#!/bin/bash
# Gates GPU du bain thermique conditionnel (T5) : EQC-1, EQC-1b, EQC-3, EQC-4 (Rust) puis, si la fibre de T6 est
# présente, EQC-4 et EQC-2 par la ligne de commande. Imprime PASS / FAIL / SKIP par gate ; RC 0 = aucune FAIL.
# (SKIP ne vaut PAS vert.)
#
# acemagic : flock -w 7200 /tmp/kevinotron_gpu.lock bash crates/kvu/validation/cond/gates_cond.sh
# gamer    : flock -w 7200 /tmp/gamer_gpu.lock KVU_FEATURES=cuda bash crates/kvu/validation/cond/gates_cond.sh
#            (fixtures générées sur acemagic, copiées : KVU_COND_FIXTURES=<dossier>)
#
# Variables : KVU_FEATURES (cuda sur le gamer), KVU_COND_FIXTURES (jeu complet L=4 et 8 ; généré s'il manque et
# si le venv numba est là), KVU_CIBLE (sinon CARGO_TARGET_DIR, sinon ../../targets/T5 du worktree), KVU_EQC2=0|1
# (EQC-2 : défaut 1 si la fibre répond), KVU_MUTANT=0|1 (mutant β×1,01 d'EQC-2, défaut 1), KVU_TMP.
# Fenêtre GPU visée : < 10 min (verrou rendu à la sortie du script).
set -u
cd "$(dirname "$0")/../../../.." || exit 2
RACINE=$(pwd)
export CARGO_TARGET_DIR="${KVU_CIBLE:-${CARGO_TARGET_DIR:-$RACINE/../../targets/T5}}"
FEAT=""; [ -n "${KVU_FEATURES:-}" ] && FEAT="--features ${KVU_FEATURES}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/cuda-13.3/lib64"
export PATH="$HOME/.cargo/bin:$PATH"
TMP="${KVU_TMP:-/tmp}/kvu_cond_gates_$$"; mkdir -p "$TMP"
LOG="$TMP/journal.txt"
trap 'rm -rf "$TMP"' EXIT
FAILS=0
dispo() { df --output=avail -BG / | tail -1 | tr -d ' G'; }
echo "== gates_cond.sh $(date +%H:%M:%S) : SSD libre $(dispo) Go, cible $CARGO_TARGET_DIR"
[ "$(dispo)" -lt 110 ] 2>/dev/null && [ "$(hostname)" = "acemagic" ] && { echo "ARRET : moins de 110 Go libres sur acemagic"; exit 7; }

# ---- 1. fixtures (jeu complet, TEMPORAIRE : L = 4 et 8, 108 cas)
FIX="${KVU_COND_FIXTURES:-}"
if [ -z "$FIX" ] || [ ! -f "$FIX/index.json" ]; then
  FIX="$TMP/fixtures"
  PY=/chemin/vers/projet/outils/ym4_noyaux/venv/bin/python
  if [ -x "$PY" ]; then
    echo "-- génération des fixtures de cb.py (jeu complet, ~1 min, numba)"
    NUMBA_NUM_THREADS=4 nice -n 10 "$PY" crates/kvu/validation/cond/gen_fixtures.py "$FIX" --niveau complet || { echo "FAIL fixtures"; exit 2; }
  else
    echo "SKIP jeu complet : pas de venv numba ici ; jeu commité (L = 4, 45 cas) seulement"; FIX=""
  fi
fi
[ -n "$FIX" ] && export KVU_COND_FIXTURES="$FIX"

# ---- 2. construction (une fois ; hors verrou si possible)
echo "-- cargo test --release $FEAT --no-run"
nice -n 10 cargo test -j 8 --release $FEAT -p kvu --test cond_gpu --test cond_cli --no-run --message-format=json > "$TMP/build.json" 2> "$TMP/build.err" \
  || { cat "$TMP/build.err"; echo "FAIL construction"; exit 2; }
exe() { python3 - "$1" "$TMP/build.json" <<'PY'
import json, sys
for l in open(sys.argv[2]):
    try: m = json.loads(l)
    except Exception: continue
    if m.get('executable') and m.get('target', {}).get('name') == sys.argv[1]:
        print(m['executable'])
PY
}
GPU_BIN=$(exe cond_gpu | tail -1); CLI_BIN=$(exe cond_cli | tail -1)
[ -x "$GPU_BIN" ] && [ -x "$CLI_BIN" ] || { echo "FAIL : binaires de test introuvables"; exit 2; }

# ---- 3. gates Rust sur le GPU : EQC-1, EQC-1b, EQC-3, EQC-4 (mécanique de reprise), β = 0, invariants du rapide
echo "-- gates GPU (cond_gpu)"
KEVINOTRON_SSD_MIN_GO="${KEVINOTRON_SSD_MIN_GO:-0}" nice -n 10 timeout 540 "$GPU_BIN" --ignored --nocapture --test-threads=1 > "$TMP/gpu.out" 2>&1
RC=$?
# (libtest imprime « test nom ... » devant la 1re ligne d'un test : on extrait à partir du mot-clé, sans ancre)
sed -n 's/.*\(PASS\|FAIL\|SKIP\|NOTE\) \(.*\)$/\1 \2/p' "$TMP/gpu.out"
[ $RC -ne 0 ] && { echo "FAIL cond_gpu (rc=$RC)"; tail -15 "$TMP/gpu.out"; FAILS=$((FAILS+1)); }
for g in "EQC-1 " "EQC-1b" "EQC-3" "EQC-4"; do
  grep -q "PASS $g" "$TMP/gpu.out" || { echo "FAIL : pas de ligne PASS $g"; FAILS=$((FAILS+1)); }
done
grep -q "≥ 100 cas" "$TMP/gpu.out" && echo "ATTENTION : EQC-1 joué sur le jeu commité seulement (< 100 cas) : gate NON complète"

# ---- 4. par la ligne de commande avec la fibre de T6 : EQC-4 (chaîne complète) puis EQC-2
cond() { KVU_COND_ARGV="$(printf '%s\x1f' "$@" | sed 's/\x1f$//')" "$CLI_BIN" --ignored --exact pont_cli --nocapture --test-threads=1; }
CFG4="$RACINE/crates/kvu/tests/donnees/cond/cfg_L4_haar0.npy"
fibre_ok=0
KEVINOTRON_SSD_MIN_GO=0 cond chaine --cfg "$CFG4" --beta 2.3 --graine 1 --mode rapide --ntherm 2 --nsweep 20 --every 2 --deach 5 --ckpt 0 \
  --out "$TMP/sonde" --sans-journal > "$TMP/sonde.out" 2>&1 && fibre_ok=1
if [ $fibre_ok -eq 1 ]; then
  for mode in ref_rng rapide; do
    A="$TMP/c4a_$mode"; B="$TMP/c4b_$mode"
    export KEVINOTRON_SSD_MIN_GO=0
    cond chaine --cfg "$CFG4" --beta 2.3 --graine 2709 --mode $mode --ntherm 5 --nsweep 60 --every 2 --deach 5 --ckpt 10 --out "$A" --sans-journal >/dev/null 2>&1
    cond chaine --cfg "$CFG4" --beta 2.3 --graine 2709 --mode $mode --ntherm 5 --nsweep 60 --every 2 --deach 5 --ckpt 10 --out "$B" --arret-apres-ckpt 2 --sans-journal >/dev/null 2>&1
    cond chaine --reprendre "$B" --sans-journal >/dev/null 2>&1
    ok=1; for f in F.npy D.npy blk.npy controles.json alea.json; do cmp -s "$A/$f" "$B/$f" || ok=0; done
    [ -f "$B/COMPLETE" ] || ok=0
    if [ $ok -eq 1 ]; then echo "PASS EQC-4 CLI ($mode) : reprise à l'octet avec la fibre réelle (nouveau processus)"; else echo "FAIL EQC-4 CLI ($mode)"; FAILS=$((FAILS+1)); fi
  done
  if [ "${KVU_EQC2:-1}" = "1" ] && [ -x /chemin/vers/projet/outils/ym4_noyaux/venv/bin/python ]; then
    echo "-- EQC-2 (rapide contre les chaînes de production L8)"
    MUT=""; [ "${KVU_MUTANT:-1}" = "1" ] && MUT="--mutant"
    NUMBA_NUM_THREADS=4 nice -n 10 /chemin/vers/projet/outils/ym4_noyaux/venv/bin/python crates/kvu/validation/cond/eqc2.py \
      --bin "$CLI_BIN" --backend auto --loi $MUT --tmp "$TMP/eqc2" --sortie "${KVU_EQC2_JSON:-$TMP/eqc2.json}" | tee "$TMP/eqc2.out" | grep -E "ROUGE|EQC-2|loi alpha"
    grep -q "^EQC-2 PASS" "$TMP/eqc2.out" && echo "PASS EQC-2 : rapide ≡ référence (seuils de §6.3), mutant ROUGE" || { echo "FAIL EQC-2"; FAILS=$((FAILS+1)); }
    [ -n "${KVU_EQC2_JSON:-}" ] && echo "(détail : $KVU_EQC2_JSON)"
  else
    echo "SKIP EQC-2 : KVU_EQC2=0 ou venv numba absent"
  fi
else
  echo "SKIP EQC-4 CLI et EQC-2 : la fibre de T6 ne répond pas encore ($(head -c 200 "$TMP/sonde.out" | tr '\n' ' '))"
fi
# ---- 5. concordance à L = 12 et 16 (cb.sweep ref_rng contre cond sweep, GPU), si le venv numba est là
if [ -x /chemin/vers/projet/outils/ym4_noyaux/venv/bin/python ] && [ -d /chemin/vers/projet/outils/ym4_noyaux/noyau2/data ]; then
  KEVINOTRON_SSD_MIN_GO=0 NUMBA_NUM_THREADS=4 nice -n 10 /chemin/vers/projet/outils/ym4_noyaux/venv/bin/python crates/kvu/validation/cond/concordance_grandL.py \
    --bin "$CLI_BIN" --backend auto | tail -3 | tee "$TMP/conc.out"
  grep -q "^PASS concordance" "$TMP/conc.out" || { echo "FAIL concordance grand L"; FAILS=$((FAILS+1)); }
else
  echo "SKIP concordance L = 12 et 16 : pas de venv numba / données noyau2 ici"
fi
echo "== bilan : $FAILS gate(s) rouge(s) ; $(date +%H:%M:%S)"
exit $([ $FAILS -eq 0 ] && echo 0 || echo 1)
